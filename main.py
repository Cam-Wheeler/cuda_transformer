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
import torch
from torch.utils.data import DataLoader
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

    print(f"Model config: {model_config}", flush=True)
    print(f"Trainer config: {trainer_config}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    # Setup the seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup the model
    model = QWEN3(model_config)

    # Setup the data.
    if args.location == "local":
        train_path = Path(os.path.dirname(__file__) + "/" + "data" + "/train.bin")
        validation_path = Path(
            os.path.dirname(__file__) + "/" + "data" + "/validation.bin"
        )
    else:
        train_path = Path("/data/train.bin")
        validation_path = Path("/data/validation.bin")

    train_dataset = TinyStoriesDataset(
        train_path, block_size=model_config.context_length, random=True
    )
    validation_dataset = TinyStoriesDataset(
        validation_path,
        block_size=model_config.context_length,
        random=False,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=False
    )  # Randomness comes from the __getitem__
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=4, shuffle=False, drop_last=True
    )

    # Setup the trainer
    trainer = Trainer(trainer_config, model, train_dataloader, validation_dataloader)

    # Train
    results = trainer.train_end_to_end()
    print(f"Results: {results}", flush=True)

    # Clean up
    return 0


if __name__ == "__main__":
    main()
