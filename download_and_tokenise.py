"""
Code for downloading and tokenising TinyStories for training. 

Creates train.bin and validation.bin within the data directory.

We are going to make use of Kaparthy's technique for tokenising the dataset.

Essentially, we take the QWEN tokeniser and tokenise each story in TinyStories.

However, doing this naively would result in a dataset where we need to pad and
do some extra work to make sure that the dataset works for training.

So here, we are generating a very simple .bin file that contains a list of ids for each story.
This avoids any issue of padding, we just slice our array as we like!

When it comes to training, we simply will simply use the standard Torch Dataset
and DataLoader combo to serve the model with data.
"""

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser


DATASET_NAME = "roneneldan/TinyStories"
NUM_PROCESSES = os.cpu_count() // 2


def main():

    parser = ArgumentParser()
    parser.add_argument("--location", type=str, default="local", choices=["local", "cluster"])
    args = parser.parse_args()

    if args.location == "cluster":
        DATA_ROOT = Path("/data")
    else:
        DATA_ROOT = Path(os.path.dirname(__file__) + "/" + "data")

    # Load in tokensier
    print("Loading Tokeniser...")
    tokeniser = AutoTokenizer.from_pretrained("QWEN/Qwen3-0.6B", cache_dir="./cache")

    # Load in dataset
    print("Loading TinyStories...")
    dataset = load_dataset(DATASET_NAME, cache_dir="./cache")

    def tokenise_row(row):
        ids = tokeniser.encode(row["text"], add_special_tokens=False)
        ids.append(tokeniser.eos_token_id)
        out = {"ids": ids, "len": len(ids)}
        return out

    print("Tokenising TinyStories...")
    # Tokenise dataset
    tokenised = dataset.map(
        tokenise_row,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=NUM_PROCESSES,
    )

    # Save to .bin files
    for split, dataset in tokenised.items():
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        arr_len = np.sum(
            dataset["len"], dtype=np.uint64
        )  # work out the total array length.
        print(f"{split}: {arr_len:,} tokens")
        filename = DATA_ROOT / f"{split}.bin"
        # We need to use a larger encoding than Kaparthy as QWEN has a much larger vocab.
        # 151,936 > 2**16 but 151,936 < 2**32
        dtype = np.uint32 
        arr = np.memmap(
            filename, dtype=dtype, mode="w+", shape=(arr_len,)
        )  # Pre allocate the array.

        idx = 0
        total_batches = 1024
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster writes.
            batch = dataset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"]).astype(dtype)
            # Write into mmap.
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        assert idx == arr_len
        arr.flush()

if __name__ == "__main__":
    main()