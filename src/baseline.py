"""
TF-IDF + logistic regression baseline for AG News.

Each article is turned into a TF-IDF vector (vocabulary and IDF are learned on
the training set only), then a logistic regression classifier is trained on top.

This is a baseline that DistilBERT has to beat. Accuracy is reported on
the validation set; the test set is scored once at the end, with a per-class
report (precision / recall / F1).

Usage:
    python -m src.baseline
"""

from src.data import split_dataset, classes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def main():
    train_set, val_set, test_set = split_dataset()
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train_set["text"])
    y_train = train_set["label"]
    x_val = vectorizer.transform(val_set["text"])
    y_val = val_set["label"]
    x_test = vectorizer.transform(test_set["text"])
    y_test = test_set["label"]
    print(x_train.shape)

    model = LogisticRegression(max_iter=1000)

    # fit the model on the training data and evaluate on the validation data
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"val_acc: {val_acc}")

    # evaluate the model on the test data
    y_test_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"test_acc: {test_acc}")
    classification_rep = classification_report(y_test, y_test_pred, target_names=classes)
    print(f"classification_report:")
    print(f"{classification_rep}")
    
    # return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    main()