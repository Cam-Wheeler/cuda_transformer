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

Papers, books and code that helped me build:
- https://github.com/rasbt/LLMs-from-scratch
- https://www.youngju.dev/blog/llm/2026-03-17-build-llm-from-scratch-guide.en#introduction
- https://github.com/karpathy/nanoGPT
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel  # V3

# local imports
from config import StandardTrainerConfig
from models.pytorch_transformer import QWEN3


class Trainer(object):
    def __init__(
        self,
        config: StandardTrainerConfig,
        model: QWEN3,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # Main training setup
        self.total_iterations = config.total_iterations
        self.warmup_iters = config.warmup_iters
        self.learning_rate = config.learning_rate  # max learning rate
        self.min_lr = (
            self.learning_rate * 0.1
        )  # should be ~= learning_rate/10 per Chinchilla
        self.optimiser = AdamW(
            params=self.model.parameters(),
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
        self.eval_interval = config.eval_interval
        self.checkpoint_after_eval = config.checkpoint_after_eval
        # Resume logic
        self.resume = config.resume
        if self.resume:
            self.resume_iter = config.resume_iter
            # perhaps we need to update the learning rate sceduler here.
        # Logs
        self.log_interval = config.log_interval
        self.wandb_log = config.wandb_log

        # Move that MF to device to go brrrrr.
        self.device = config.device
        self.model.to(self.device)

    def train_end_to_end(self):
        """
        Runs a full training loop.
        """

        # number of iterations

        # run single training mini-batch

        # step the learning rate scheduler

        # when eval, validate

        # collate the results

        # log

        # checkpoint if we did well.
        pass

    def train_single(self):
        """
        Trains a single mini-batch.
        """

        # for the number of batches in epoch

        # forward

        # loss

        # backward
        pass

    def validate_single(self):
        """
        Validates on the validation set a single time.
        """

        # model into eval mode

        # run eval

        # collect metrics
        pass

    def log_metrics(self, metrics, wandb=False):
        """
        Logs metrics locally and to wandb.
        """

        # log locally

        # log to wandb

        pass

    def save_checkpoint(self):
        """
        Saves checkpoints.
        """

        # checkpoint the model, this will only be called when we do better
        # on the validation set.
        pass
