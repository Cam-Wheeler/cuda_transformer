"""
Complete end to end pipeline:
    1. Read in args for training.
    2. Setup configs using args.
    3. Setup the model.
    4. Setup the trainer.
    5. Train.
    6. Clean up.
    7. Party 🎉
"""

import os
import argparse
from typing import Tuple, Union
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# Local imports
from training.config import (
    QWEN3_MINI_Config,
    QWEN3_06B_Config,
    MiniTrainerConfig,
    StandardTrainerConfig,
)
from training.train import Trainer
from models.pytorch_transformer import QWEN3
from training.dataset import TinyStoriesDataset

MODEL_CONFIGS = {
    "qwen3_mini": QWEN3_MINI_Config(),
    "qwen3_06b": QWEN3_06B_Config(),
}

TRAINER_CONFIGS = {
    "mini": MiniTrainerConfig(),
    "standard": StandardTrainerConfig(),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", type=str, required=True, choices=MODEL_CONFIGS.keys()
    )
    parser.add_argument(
        "--trainer_config", type=str, required=True, choices=TRAINER_CONFIGS.keys()
    )
    parser.add_argument(
        "--location", type=str, required=True, choices=["local", "cluster"]
    )
    parser.add_argument("--seed", type=int, default=1501)

    args = parser.parse_args()

    # Setup the configs
    model_config = MODEL_CONFIGS[args.model_config]
    trainer_config = TRAINER_CONFIGS[args.trainer_config]

    # Configure the training environment for DDP or single process training.
    ddp, rank, local_rank, world_size, device = configure_training_environment()

    try: # Once ddp is setup, we need to tear it down properly. 
        
        setup_seed(args.seed, ddp, rank)

        # If DDP rank must be 0 to be master process. if not ddp we dont care.
        master_process = rank == 0
        if master_process:
            print(f"Model config: {model_config}", flush=True)
            print(f"Trainer config: {trainer_config}", flush=True)
            print(f"Seed: {args.seed}", flush=True)

        # Setup the data.
        # We are not using a DistributedSampler here even on DDP, we are following
        # Kaparthy's approach of just altering the seed for each process!
        train_path, validation_path = get_train_and_validation_paths(args.location)
        train_dataloader, validation_dataloader = create_dataloaders(
            train_path,
            validation_path,
            model_config.context_length, 
            trainer_config.batch_size,
        )

        # Setup the model
        model = setup_model(model_config, ddp, local_rank, device)

        # Setup the trainer
        gradient_accumulation_steps = update_gradient_accumulation_steps(
            trainer_config.gradient_accumulation_steps, ddp, world_size,
        )
        trainer = Trainer(
            trainer_config,
            model,
            train_dataloader,
            validation_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ddp=ddp,
            world_size=world_size,
            master_process=master_process,
            device=device
        )

        # Train
        results = trainer.train_end_to_end()
        if master_process:
            print(f"Results: {results}", flush=True)

    finally:
        destroy_training_environment(ddp)
    return 0

# Helpers to clean up main a bit.
def configure_training_environment() -> Tuple[bool, int, int, int, str]:  # ddp, rank, local_rank, world_size, device
    """
    Configure the training environment for DDP or single process training.
    """
    ddp = int(os.environ.get("RANK", -1)) != -1  # Check if DDP is enabled.
    if not ddp:
        return False, 0, 0, 1, "cuda:0"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    return True, int(os.environ["RANK"]), local_rank, world_size, device

def setup_seed(seed: int, ddp: bool, rank: int) -> None:
    """
    Setup the seed for the training environment. This is really important
    for DDP as we stagger the seeds for each GPU. If we dont do this
    each GPU will have the same see and training will be identical for each GPU.
    """
    seed_offset = 0
    if ddp:
        seed_offset = rank
    torch.manual_seed(seed + seed_offset)  # Offset each GPU when in DDP.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + seed_offset)

def get_train_and_validation_paths(location: str) -> Tuple[Path, Path]:
    """
    Get the train and validation paths from the data directory.
    """
    if location == "local":
        return (
            Path(__file__).parent / "data" / "train.bin",
            Path(__file__).parent / "data" / "validation.bin",
        )
    else:
        return (Path("/data/train.bin"), Path("/data/validation.bin"))

def create_dataloaders(
    train_path: Path,
    validation_path: Path,
    block_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create the dataloaders for the training and validation datasets and dataloaders.
    Randomness still comes from the __getitem__ method of the dataset.
    """
    train_dataset = TinyStoriesDataset(
        train_path, block_size=block_size, random=True,
    )
    validation_dataset = TinyStoriesDataset(
        validation_path, block_size=block_size, random=False
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_dataloader, validation_dataloader

def setup_model(
    model_config: Union[QWEN3_MINI_Config, QWEN3_06B_Config],
    ddp: bool,
    local_rank: int,
    device: str,
) -> QWEN3:
    """
    Setup the model for DDP or single process training.
    """
    model = QWEN3(model_config).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model

def update_gradient_accumulation_steps(gradient_accumulation_steps: int, ddp: bool, world_size: int) -> int:
    """
    Update the gradient accumulation steps for training. This is required for
    DDP as we do not want to scale the tokens seen per step by the world size.
    This keeps the number of tokens seen while accumulating gradients per step the same.
    """
    if ddp:
        assert gradient_accumulation_steps % world_size == 0, (
            "Gradient accumulation steps must be divisible by world size"
        )
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    return gradient_accumulation_steps

def destroy_training_environment(ddp: bool) -> None:
    """
    Destroy the training environment for DDP or single process training.
    """
    if ddp:
        dist.destroy_process_group()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
