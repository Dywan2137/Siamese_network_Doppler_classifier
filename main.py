"""Run the whole thing: load -> train -> embed -> classify -> plot.

    python main.py

This file is the order of the work and nothing else. Read it top to bottom to
see what the project does; follow an import to see how a step works.
"""

import torch

import config as cfg
import classify
import data
import plots
from model import build_model
from train import embed, train_model


def main():
    device = torch.device("cpu")
    print("Device:", device)

    # data
    print("\n=== 1. DATA ===")
    train_ds, test_ds, train_loader, test_loader = data.load_split()

    # model
    print("\n=== 2. MODEL ===")
    model = build_model(device)

    # train
    print("\n=== 3. TRAIN ===")
    history = train_model(model, train_loader, device)
    plots.plot_loss(history)

    # embed
    print("\n=== 4. EMBED ===")
    X_train, y_train = embed(model, train_loader, device)
    X_test, y_test = embed(model, test_loader, device)
    print(f"Train embeddings: {X_train.shape}   Test embeddings: {X_test.shape}")

    # classify
    print("\n=== 5. CLASSIFY (KNN) ===")
    preds, acc, cm = classify.knn_classify(X_train, y_train, X_test, y_test)

    # check with the random baseline
    print("\n=== 6. RANDOM BASELINE ===")
    classify.random_baseline(train_loader, test_loader, acc, device)

    # plots
    print("\n=== 7. PLOTS ===")
    plots.plot_confusion(cm, acc)
    plots.plot_embedding_space(X_train, y_train, X_test, y_test)
    plots.plot_label_grid(train_ds, "Train set sample", "train_grid.png")
    plots.plot_label_grid(test_ds, "Test set sample", "test_grid.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
