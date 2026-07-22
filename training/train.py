"""
Code for training the LLM from scratch

While QWEN3 does not specifically chat about their training regieme there
are some things we can extract from previous papers / repos:
    - Loss: autoregressive cross-entropy (V1)
    - Optimiser: AdamW (V1)
    - Weight decay (not for bias / norms) (V2)
    - Gradient clipping (V2)
    - LR schedule: Linear warmup with cosine decay (V1)

We are also going to use Distributed training (DDP) in our training (V3).
We will add support for resuming training in (V2).

Papers, books and code that helped me build:
- https://github.com/rasbt/LLMs-from-scratch
- https://www.youngju.dev/blog/llm/2026-03-17-build-llm-from-scratch-guide.en#introduction
- https://github.com/karpathy/nanoGPT
"""

import wandb
import torch
from pathlib import Path
from typing import Dict, Tuple, Iterator, Optional, Union
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
from torch import distributed as dist
from contextlib import nullcontext
# local imports
from training.config import StandardTrainerConfig, MiniTrainerConfig
from models.pytorch_transformer import QWEN3

class Trainer(object):
    def __init__(
        self,
        config: Union[StandardTrainerConfig, MiniTrainerConfig],
        model: QWEN3,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        gradient_accumulation_steps: int,
        ddp: bool,
        world_size: int,
        master_process: bool,
        device: str    
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.device = device

        # Main training setup
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.total_iterations: int = config.total_iterations
        self.warmup_iters: int = config.warmup_iters
        self.learning_rate: float = config.learning_rate  # max learning rate
        self.min_lr: float = (
            self.learning_rate * 0.1
        )  # should be ~= learning_rate/10 per Chinchilla
        self.optimiser = AdamW(
            params=self._select_param_groups(
                weight_decay=1e-1
            ),  # Selective weight decay, ignore biases and norms.
            lr=self.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.lr_warmup = LinearLR(
            optimizer=self.optimiser,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.warmup_iters,
        )
        self.lr_decay = CosineAnnealingLR(
            optimizer=self.optimiser,
            T_max=self.total_iterations - self.warmup_iters,
            eta_min=self.min_lr,
        )
        self.lr_scheduler = SequentialLR(
            self.optimiser,
            schedulers=[self.lr_warmup, self.lr_decay],
            milestones=[self.warmup_iters],
        )
        self.eval_interval: int = config.eval_interval

        # Resume logic
        self.resume: bool = config.resume
        if self.resume:
            # TODO: implement resume logic
            # perhaps we need to update the learning rate sceduler here.
            pass

        # Logs and saving
        self.log_interval: int = config.log_interval
        self.root_save_path: Path = Path(config.root_save_path)
        self.wandb_log: bool = config.wandb_log
        if self.wandb_log:
            self.entity = config.wandb_entity
            self.project = config.wandb_project

        # DDP flags
        self.ddp: bool = ddp
        self.master_process: bool = master_process
        self.world_size: int = world_size

    def train_end_to_end(self) -> Dict[str, int]:
        """
        Runs a full training loop.
        """
        total_tokens_seen = 0
        train_loss = 0
        validation_loss = 0
        validation_runs = 0

        if self.wandb_log and self.master_process:
            wandb.init(
                entity=self.entity,
                project=self.project,
                config=self.config.model_dump(),
            )
        if self.ddp:
            dist.barrier() # Processes will wait while the master sets up logging.

        try:
            # number of iterations
            for i in range(self.total_iterations):
                # Iterate through the mini-batch.
                self.optimiser.zero_grad()
                train_results = self._train_single()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimiser.step()
                total_tokens_seen += train_results["tokens"]
                train_loss += train_results["loss"]

                # step the learning rate scheduler
                self.lr_scheduler.step()

                # when eval, validate
                validation_results = None
                if i % self.eval_interval == 0:
                    if self.master_process:
                        validation_results = self._validate_single()
                        validation_loss += validation_results["loss"]
                        validation_runs += 1
                        self._save_checkpoint(
                            i, validation_results["loss"], total_tokens_seen
                        ) # Checkpoint at every eval.
                    if self.ddp:
                        dist.barrier() # Processes will wait while we eval on the master process.

                # log:
                if self.master_process and i % self.log_interval == 0:
                    self._log_metrics(
                        i,
                        total_tokens_seen,
                        train_results,
                        validation_results,
                        use_wandb=self.wandb_log,
                    ) # no need for a barrier for logging.
        finally:
            if self.master_process and self.wandb_log:
                wandb.finish()
            if self.ddp:
                dist.barrier() # Everyone wait while we teardown wandb.

        return {
            "total_tokens": total_tokens_seen,
            "average_train_loss": train_loss / self.total_iterations,
            "average_validation_loss": (
                validation_loss / validation_runs if validation_runs > 0 else None
            ),
        }

    def _train_single(self) -> Dict[str, float]:
        """
        Trains over a self.gradient_accumulation_steps of mini-batches.
        """
        self.model.train()
        training_stats = torch.zeros(2, device=self.device) # mini_batch_loss, tokens_seen

        # Iterate over the gradient accumulation steps and run a forward and backward each time.
        for mini_batch_idx in range(self.gradient_accumulation_steps):
            (input, target), self.train_iter = self._get_batch(
                self.train_dataloader, self.train_iter
            )
            input, target = input.to(self.device), target.to(self.device)

            #  We do not want to sync until the last step in the mini-batch.
            if self.ddp and (mini_batch_idx + 1) != self.gradient_accumulation_steps:
                context_manager = self.model.no_sync() # No syncing.
            else:
                context_manager = nullcontext() # empty context in ddp it will allow syncing, if not ddp, nothing happens.

            with context_manager:
                logits = self.model(
                    input
                )  # [batch, seq_len] in, [batch, seq_len, vocab_size] out
                loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                training_stats[0] += loss.detach()
                training_stats[1] += input.numel()
            
        # all reduce (sum) on the metrics from this batch.
        if self.ddp:
            dist.all_reduce(training_stats, op=dist.ReduceOp.SUM)
            training_stats[0] /= self.world_size # make the loss the average across GPUS.

        return {"loss": training_stats[0].item(), "tokens": int(training_stats[1].item())}

    def _get_batch(self, data_loader, iterator) -> Tuple[torch.tensor, Iterator]:
        """
        Returns a single batch for a forward, resets when we exhaust the dataset.
        """
        try:
            batch = next(iterator)  # try next batch
            return batch, iterator  # hand back batch + same iterator
        except StopIteration:
            iterator = iter(data_loader)  # epoch finished → restart
            batch = next(iterator)  # take first batch of new epoch
            return batch, iterator  # return batch and the new iterator!

    def _validate_single(self) -> Dict[str, float]:
        """
        Validates on the validation set a single time.
        """
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            for input, target in self.val_dataloader:
                input, target = input.to(self.device), target.to(self.device)
                logits = self.model(input)
                loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
                valid_loss += loss.item()
        return {
            "loss": valid_loss / len(self.val_dataloader)  # Average validation loss.
        }

    def _log_metrics(
        self,
        step: int,
        tokens_seen_so_far: int,
        train_metrics: Dict[str, float],
        validation_metrics: Optional[Dict[str, float]] = None,
        use_wandb: bool = False,
    ) -> None:
        lr = self.optimiser.param_groups[0]["lr"]

        metrics = {
            "train/loss": train_metrics["loss"],
            "train/tokens_seen": tokens_seen_so_far,
            "lr": lr,
        }
        if validation_metrics is not None:
            metrics["val/loss"] = validation_metrics["loss"]

        # local
        print(f"step {step}: {metrics}", flush=True)

        if use_wandb:
            wandb.log(metrics, step=step)

    def _save_checkpoint(
        self, step: int, val_loss: float, total_tokens_seen: int
    ) -> None:
        path = self.root_save_path / "checkpoints" / f"step_{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        model_checkpoint = self.model.module.state_dict() if self.ddp else self.model.state_dict()
        checkpoint = {
            "step": step,
            "total_tokens_seen": total_tokens_seen,
            "val_loss": val_loss,
            "model": model_checkpoint,
            "optimiser": self.optimiser.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config.model_dump(),
        }
        torch.save(checkpoint, path)
        print(f"saved checkpoint → {path}", flush=True)

    def _select_param_groups(self, weight_decay: float = 1e-1):
        """
        Selects the param groups for the optimiser.
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or name.endswith(".scale"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
