"""

Code for downloading the Tiny Stories dataset.

Run this script on its own to download the dataset.

```bash
python3 tiny_stories.py
```
"""

from datasets import load_dataset
from dotenv import load_dotenv

DATASET_NAME = "roneneldan/TinyStories"

if __name__ == "__main__":
    load_dotenv() # Load environment variables from .env file, it will download regardless.
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, cache_dir="./cache")
    print(dataset)
