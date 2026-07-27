"""Classify embeddings with KNN, and check how much the training mattered.

knn_classify: the actual classifier — a K-nearest-neighbours vote in
embedding space.

random_baseline: the honesty check. It runs the SAME KNN on embeddings
from an untrained, randomly-initialised network. If a random network
already scores high, the task is easy for KNN and the Siamese training is
barely doing any work.
  
Fun fact the random baseline has already high accuracy which means the task is too easy
for siamese network, but i still wanted to do it.

"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import config as cfg
from train import embed


def knn_classify(X_train, y_train, X_test, y_test, k=cfg.KNN_NEIGHBORS):
    """Fit KNN on train embeddings, score on test. Returns (preds, acc, cm)."""
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=list(range(cfg.NUM_CLASSES)))
    print(f"Test accuracy: {acc * 100:.1f}%")
    return preds, acc, cm


def random_baseline(train_loader, test_loader, trained_acc,
                    device="cpu", k=cfg.KNN_NEIGHBORS):
    """KNN accuracy using an UNTRAINED network — the "is training necessary?" test.

    Returns the random-model accuracy so it can be reported next to the trained
    one.
    """
    from model import build_model

    random_model = build_model(device)   # fresh random weights, never trained
    Xtr, ytr = embed(random_model, train_loader, device)
    Xte, yte = embed(random_model, test_loader, device)

    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    random_acc = knn.score(Xte, yte)

    print(f"Trained model KNN accuracy : {trained_acc * 100:.1f}%")
    print(f"Random  model KNN accuracy : {random_acc * 100:.1f}%")
    if random_acc > 0.9:
        print("A random network already exceeds 90% — the task is THIS easy for KNN.")
        print("The Siamese training is barely necessary for this data.")
    else:
        print(f"Random network manages only {random_acc * 100:.1f}% — training helped.")

    return random_acc
