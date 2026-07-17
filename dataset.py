"""
Torch Dataset for TinyStories .bin files

This needed to be pached a bit as torch uses __len__ to determine the shuffled
indices which was causing OOM. So we now use a random index to get a block of data.

DataLoader shuffle builds a permutation of size __len__, which OOMed when __len__ was ~every token position.
__len__ is now samples-per-epoch, not “number of possible windows in the file."
__getitem__ ignores idx as a memmap offset and samples a random start for each block.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path
import numpy as np


class TinyStoriesDatset(Dataset):

    def __init__(self, dataset: Path, block_size: int, random: bool = True) -> None:
        super().__init__()
        assert dataset.is_file(), "Dataset does not exist at path."
        self.data = np.memmap(dataset, dtype=np.uint32, mode="r")
        self.block_size = block_size
        self.random = random

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random:
            i = torch.randint(0, len(self.data) - self.block_size, size=(1,)).item()
        else:
            # Non-overlapping windows: idx=0 -> start 0, idx=1 -> start block_size, ...
            i = idx * self.block_size

        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + 1 + self.block_size]

        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64)),
        )
