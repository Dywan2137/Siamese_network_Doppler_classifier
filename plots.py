"""
Figures: embedding space, confusion matrix, and label sanity-check grids.

"""

import os
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import config as cfg

# colors for each class 
_CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#dc143c", "#1e90ff", "#2f2f2f",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#3cb371",
]


def _save(fig, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")


def plot_confusion(cm, acc, save_dir=cfg.OUTPUT_DIR):
    """Confusion matrix heatmap with class names."""
    names = [cfg.IDX_TO_LABEL[i] for i in range(cfg.NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test acc = {acc * 100:.1f}%)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, "confusion_matrix.png", save_dir)
    plt.close(fig)


def plot_embedding_space(X_train, y_train, X_test, y_test, save_dir=cfg.OUTPUT_DIR):
    """
    t-SNE projection of the embeddings; train = dots, test = stars.
    """
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    is_test = np.concatenate([np.zeros(len(X_train), bool),
                              np.ones(len(X_test), bool)])

    perplexity = min(15, max(2, len(X_all) // 3))   # guard tiny sets
    X_2d = TSNE(n_components=2, perplexity=perplexity,
                random_state=cfg.SEED, init="pca").fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(13, 10))
    for c in range(cfg.NUM_CLASSES):
        color = _CLASS_COLORS[c % len(_CLASS_COLORS)]
        tr = (y_all == c) & ~is_test
        te = (y_all == c) & is_test
        ax.scatter(X_2d[tr, 0], X_2d[tr, 1], c=color, s=70, alpha=0.7,
                   edgecolors="white", linewidths=0.8, label=cfg.IDX_TO_LABEL[c])
        ax.scatter(X_2d[te, 0], X_2d[te, 1], c=color, marker="*", s=200,
                   edgecolors="black", linewidths=1.2)

    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_title("Doppler embedding space (t-SNE)\n● train   ★ test")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "embedding_space.png", save_dir)
    plt.close(fig)


def plot_label_grid(dataset, title, name, per_class=5, save_dir=cfg.OUTPUT_DIR):
    """
    Grid of images with their labels for manual checks.
    """
    labels = [dataset[i][1] for i in range(len(dataset))]

    picks = []
    for c in range(cfg.NUM_CLASSES):
        idxs = [i for i, l in enumerate(labels) if l == c][:per_class]
        picks.extend(idxs)

    cols = per_class or 5
    rows = max(1, (len(picks) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, i in zip(axes, picks):
        img, lbl = dataset[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"[{lbl}] {cfg.IDX_TO_LABEL[lbl]}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")

    plt.suptitle(title, y=1.0)
    plt.tight_layout()
    _save(fig, name, save_dir)
    plt.close(fig)

    print(f"\n{title} — label distribution:")
    for lbl, cnt in sorted(Counter(labels).items()):
        print(f"  [{lbl}] {cfg.IDX_TO_LABEL[lbl]:<25}  {cnt} images")


def plot_loss(history, save_dir=cfg.OUTPUT_DIR):
    """
    Training-loss curve.
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history) + 1), history)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Triplet loss")
    ax.set_title("Training loss"); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "training_loss.png", save_dir)
    plt.close(fig)
