"""
Fine-tune DistilBERT on AG News.

Steps:
    1. Prepare the data (from src/data.py)
    2. Build the model (from src/model.py)
    3. Set up the optimizer, learning rate and scheduler
    4. Training loop: compute loss -> backward pass -> update weights
    5. Evaluate on the validation set after each epoch and save the best model
    6. Evaluate on the test set once at the end

Usage:
    python -m src.train
"""

import os
import torch
import random
import numpy as np
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from src.data import split_dataset, tokenize_dataset, model_name, data_path, max_length
from src.model import TextClassifier

batch_size = 32
lr=5e-5
epochs=3
seed = 42

def set_seed(seed: int, deterministic: bool = True) -> None:
    # Seed every source of randomness: Python, NumPy, PyTorch on CPU and on all GPUs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GPU only: deterministic cuDNN kernels make results repeatable at some cost in speed
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def get_dataloaders(batch_size):
    train_set, val_set, test_set = split_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tokenized = tokenize_dataset(train_set, tokenizer)
    val_tokenized = tokenize_dataset(val_set, tokenizer)
    test_tokenized = tokenize_dataset(test_set, tokenizer)

    # As the texts have different lengths, we need to pad them to the same length in a batch. 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_tokenized, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, device, train, optimizer=None, scheduler=None):
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    for idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad() # clear the history gradient
            loss.backward() # compute the gradients of the loss with respect to the model parameters
            optimizer.step() # update the model parameters (weights)
            scheduler.step() # tuning the learning rate

            if idx % 100 == 0:
                print(f"Step {idx}/{len(loader)}: loss {loss.item():.4f}")
            
        total_loss += loss.item() # we get the average loss of the batch
        preds = logits.argmax(dim=1) # index of the highest score in each row = predicted class
        correct += (preds == labels).sum().item() 
        total += labels.size(0)
    return total_loss / len(loader), correct / total # len(loader) is the number of batches


def save_metrics(params, history, best_val_acc, test_loss, test_acc):
    results = {
        "params": params,
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    with open(os.path.join(data_path, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


def train():
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextClassifier().to(device)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # len(train_loader): the number of batches in the training set
    # update the learning rate once per batch
    # so here, we compute the total number of updating lr
    total_steps = epochs * len(train_loader)

    # standard learning rate schedule for fine-tuning BERT
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    history = []
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, device, train=True,
                                            optimizer=optimizer, scheduler=scheduler)
        val_loss, val_acc = run_epoch(model, val_loader, device, train=False)
        print(f"epoch {epoch + 1}: train_loss {train_loss:.4f}, train_acc {train_acc:.4f}, "
                f"val_loss {val_loss:.4f}, val_acc {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(data_path, "model.pt"))
            print(f"Best model saved with val_acc {best_val_acc:.4f}")

    model.load_state_dict(torch.load(os.path.join(data_path, "model.pt")))
    test_loss, test_acc = run_epoch(model, test_loader, device, train=False)
    print(f"test_loss {test_loss:.4f}, test_acc {test_acc:.4f}")

    params = {
        "model_name": model_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
    }
    save_metrics(params, history, best_val_acc, test_loss, test_acc)
    print(f"metrics saved to {os.path.join(data_path, 'metrics.json')}")



if __name__ == "__main__":
    train()
