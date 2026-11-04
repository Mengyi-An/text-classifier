"""
DistilBERT text classifier for AG News.

Structure:
    1. Encoder: pretrained DistilBERT turns each token into a 768-dim vector
    2. Take the [CLS] vector as the representation of the whole text
    3. Classifier: Dropout + Linear(768, 4) outputs logits for the 4 classes

The model outputs logits only, as softmax is included in CrossEntropyLoss.

Usage (shape check):
    python -m src.model
"""

import torch
from torch import nn
from transformers import AutoModel
from src.data import model_name, classes

class TextClassifier(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        # load model from HuggingFace, then save it as self.encoder (DistilBertModel)
        self.encoder = AutoModel.from_pretrained(model_name)

        # get the length of output vector
        hidden_size = self.encoder.config.hidden_size

        # classifier
        self.classifier = nn.Sequential(
            # avoid overfitting
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(classes)) # softmax is included in CrossEntropyLoss, so we don't need to add it here
        )
    
    def forward(self, input_ids, attention_mask):
        # convert token into vector
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # get the first token in the whole sequence
        cls_vector = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_vector)

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 编码两句话，padding 补齐成一样长，return_tensors="pt" 转成 PyTorch tensor
    batch = tokenizer(["Apple reports record profit", "Team wins the final"],
                    padding=True, return_tensors="pt")

    model = TextClassifier()
    logits = model(batch["input_ids"], batch["attention_mask"])
    print(logits.shape)