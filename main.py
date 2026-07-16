"""
Complete end to end pipeline:
    X. Read in args for training.
    X. Setup configs using args.
    X. Setup the model.
    X. Setup the trainer.
    X. Train.
    X. Clean up.
    X. Party 🎉
"""

import argparse
import torch
from torch.utils.data import DataLoader

# Local imports
from config import (
    QWEN3_MINI_Config,
    QWEN3_06B_Config,
)
from config import (
    StandardTrainerConfig,
)
from train import Trainer
from models.pytorch_transformer import QWEN3

MODEL_CONFIGS = {
    "qwen3_mini": QWEN3_MINI_Config(),
    "qwen3_06b": QWEN3_06B_Config(),
}

TRAINER_CONFIGS = {
    "standard": StandardTrainerConfig(),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--trainer_config", type=str, required=True, choices=TRAINER_CONFIGS.keys())
    parser.add_argument("--seed", type=int, default=1501)

    args = parser.parse_args()

    # Setup the configs
    model_config = MODEL_CONFIGS[args.model_config]
    trainer_config = TRAINER_CONFIGS[args.trainer_config]
    
    print(f"Model config: {model_config}")
    print(f"Trainer config: {trainer_config}")
    print(f"Seed: {args.seed}")

    # Setup the seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup the model
    model = QWEN3(model_config)

    # Setup the data.
    # We will need to sort out the data loaders here.

    # Setup the trainer
    # trainer = Trainer(trainer_config, model)

    # Train
    # trainer.train()

    # Clean up
    return 0

if __name__ == "__main__":
    main()