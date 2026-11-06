"""
Predict the class of news texts with the trained DistilBERT model.
Usage:
    python -m src.predict
"""


import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.data import data_path, model_name, classes, max_length
from src.model import TextClassifier

model_path = os.path.join(data_path, "model.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = TextClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model


def predict(texts, model, tokenizer):
    """
    function: predict the class of each text and how confident the model is
        1. tokenize the texts (pad to the same length, truncate to max_length)
        2. run the model to get logits
        3. softmax turns logits into probabilities; the highest one is the confidence

    input:
        texts: list[str], e.g. ["Team wins the final", "Apple reports record profit"]
    output:
        [{"class": "Sports", "confidence": 0.9987}, {"class": "Business", "confidence": 0.8702}]
    """
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probas = F.softmax(logits, dim=1) # the output of probas: tensor([[8.0416e-01, 1.9455e-01, 1.0554e-03, 2.3612e-04]])
        confidences, indices = probas.max(dim=1) # max instead of argmax, as we should get confidence and index. 

    return [{"class": classes[i], "confidence": round(p, 4)} for i, p in zip(indices.tolist(), confidences.tolist())]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model()

    texts = [
        "Team wins the final after a late goal"
    ]
    probas = predict(texts, model, tokenizer)
    print(f"probas {probas}")
