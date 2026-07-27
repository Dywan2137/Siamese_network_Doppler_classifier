"""Train the embedding network, and turn a DataLoader into embeddings.

The loss falls quickly; a handful of epochs is enough. More training does not
meaningfully change the classifier because the task is too easy for KNN and the 
embeding space shows that each class cluster is far away from the others.
"""

import numpy as np
import torch

import config as cfg
from losses import triplet_hard_loss


def train_model(model, train_loader, device="cpu",
                epochs=cfg.EPOCHS, lr=cfg.LR, margin=cfg.MARGIN):
    """
    Train with Adam + batch-hard triplet loss. Returns the loss history.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            emb = model(imgs)
            loss = triplet_hard_loss(emb, labels, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)
        history.append(avg)
        print(f"Epoch {epoch:>3}/{epochs}   loss = {avg:.4f}")

    return history


@torch.no_grad()
def embed(model, loader, device="cpu"):
    """
    Run a loader through the model and stack the embeddings + labels.
    """
    model.eval()
    embs, labels = [], []
    for imgs, lbl in loader:
        embs.append(model(imgs.to(device)).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(embs), np.concatenate(labels)
