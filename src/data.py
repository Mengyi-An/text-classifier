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
from transformers import AutoTokenizer

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
dataset_name = "fancyzhx/ag_news"
classes = ["World", "Sports", "Business", "Sci/Tech"]
model_name = "distilbert-base-uncased"
max_length = 128

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


def describe_split(name, dataset):
    """
    function: use split() to split text
    output:
        words per text: mean 37.9, max 177
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

def describe_tokens(name, dataset, tokenizer):
    """
    function: 
    1. count the number of tokens in each text
    2. use Autotokenizer to split text
    3. prepare for DistilBERT model

    output:
        tokens per text: mean 53.2, max 379
    """
    n_tokens = []
    for text in dataset["text"]:
        # tokens is a dict 
        # output: 
        # {'input_ids': [...], # token 的编号
        # 'token_type_ids': [...],  # 标记每个 token 属于第几段文本
        # 'attention_mask': [...] # 这个位置是不是真实的 token
        # }
        tokens = tokenizer(text) 
        num_tokens = len(tokens["input_ids"])
        n_tokens.append(num_tokens)
    n_tokens = np.array(n_tokens) # numpy
    # print(f"{name} n_tokens: {n_tokens}") in order to check the type of n_tokens
    for limit in [64, 128]:
        print(f"  {name} longer than {limit}: {(n_tokens > limit).mean():.2%}")

def tokenize(batch, tokenizer):
    """
    function: use tokenizer to split each batch into tokens
    """
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def tokenize_dataset(dataset, tokenizer):
    """
    function: split the whole dataset into tokens, and call tokenize() to split each batch into tokens
    """
    # dataset.map() is similar with for loop, which could be used to deal with each batch of data in the dataset.
    return dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text"],
    )


if __name__ == "__main__":
    train_set, val_set, test_set = split_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    describe_tokens("training", train_set, tokenizer)
    describe_tokens("validation", val_set, tokenizer)
    describe_tokens("test", test_set, tokenizer)

    train_tokenized = tokenize_dataset(train_set, tokenizer)
    print(train_tokenized)
    print(train_tokenized[0])

    