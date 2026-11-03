import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from src.data import split_dataset, tokenize_dataset, model_name

batch_size = 32

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


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() # we get the average loss of the batch
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct/total # len(loader) is the number of batches



if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    print(f"train_loader {type(train_loader)}")
    print(f"val_loader {type(val_loader)}")
    print(f"test_loader {type(test_loader)}")
