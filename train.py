"""
Code for training the LLM from scratch

While QWEN3 does not specifically chat about their training regieme there
are some things we can extract from previous papers / repos:
    - Loss: autoregressive cross-entropy
    - Optimiser: AdamW
    - Weight decay (not for bias / norms)
    - Gradient clipping
    - Precision: bf16
    - LR schedule: Linear warmup with cosine decay.
    - Warm up schedule: 0.1% to 20% of our total training steps.

We are also going to use Distributed training (DDP) in our training.

Papers, books and code that helped me build:
- https://github.com/rasbt/LLMs-from-scratch
- https://www.youngju.dev/blog/llm/2026-03-17-build-llm-from-scratch-guide.en#introduction
- https://github.com/karpathy/nanoGPT
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

# local imports
from config import TrainingConfig
from models.pytorch_transformer import QWEN3


class Trainer(object):

    def __init__(self, config: TrainingConfig, model: QWEN3, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


    def train_end_to_end(self):
        """
        Runs a full training loop.
        """

        # number of iterations

        # run single training

        # when eval, eval

        # collate the results

        # log

        # checkpoint if we did well. 
        pass

    def train_single(self):
        """
        Trains a single epoch.
        """

        # for the number of batches in epoch

        # forward

        # loss

        # backward

        # step
        pass

    def validate_single(self):
        """
        Validates on the validation set.
        """

        # model into eval mode

        # run eval

        # collect metrics
        pass

    def get_learning_rate(self):
        """
        Computes the learning rate.
        """

        # need to figure out what lr schedule we are going to use.

        # compute the new lr. 

        # return it.
        pass

    def log_metrics(self):
        """
        Logs metrics to wandb.
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