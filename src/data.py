"""
Load datasets "AGNews", which is downloaded from the Hugging Face Hub on first run (~30MB).
4 classes (World / Sports / Business / Sci/Tech), 120k train and 7.6k test articles.

The 120k training articles are split into 115k train / 5k validation (fixed split).
The best model is picked on validation; the test set is used exactly once at the end.

Usage (print dataset statistics):
    python -m src.data

"""



import os
import numpy as np
from datasets import load_dataset

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
dataset_name = "fancyzhx/ag_news"
classes = ["World", "Sports", "Business", "Sci/Tech"]

val_size = 5000
split_seed = 42




def split_dataset():
    """
    function: split dataset into train, val, test

    output: 
    train: Dataset({
            features: ['text', 'label'],
            num_rows: 115000
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 5000
        })
    })
    """
    raw_dataset = load_dataset(dataset_name, cache_dir=data_path)
    
    # Split the training set into train and validation sets
    split = raw_dataset["train"].train_test_split(
        test_size=val_size, seed=split_seed
    )
    test_set = raw_dataset["test"]
    train_set = split["train"]
    val_set = split["test"]

    return train_set, val_set, test_set


def describe(name, dataset):
    """
    function: count the number of words in each text, as well as the class labels

    output:
        words per text: mean 43.2, max 500
        class: [0 1 2 3]
    """
    labels = dataset["label"]
    n_words = []
    for text in dataset["text"]:
        words = text.split()
        num_words = len(words)
        n_words.append(num_words)
    n_words = np.array(n_words)
    print(f" {name} words per text: mean {n_words.mean():.1f}, max {n_words.max()}")
    counts = np.bincount(labels)
    print(f" {name} class: {dict(zip(classes, counts.tolist()))}")


if __name__ == "__main__":
    train_set, val_set, test_set = split_dataset()
    describe("training", train_set)
    describe("validation", val_set)
    describe("test", test_set)
    print("training set:", len(train_set), "\nval set:", len(val_set), "\ntest set:", len(test_set))   # 期望看到：115000 5000 7600 