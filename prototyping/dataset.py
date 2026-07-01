"""Working with the Tiny Stories dataset."""


from datasets import load_dataset
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    dataset = load_dataset("roneneldan/TinyStories", cache_dir="../data/cache")

    print("Dataset:")
    print(dataset)

    print("\n")

    print("Dataset Info:")
    print(dataset.keys())
    print("Training set length: ", len(dataset["train"]))
    print("Validation set length: ", len(dataset["validation"]))
    print("\n")


    print("Training set:")
    for example in dataset["train"]:
        print(example["text"])
        print("-" * 100)
        break

    print("\n")

    print("Validation set:")
    for example in dataset["validation"]:
        print(example["text"])
        print("-" * 100)
        break

    
        