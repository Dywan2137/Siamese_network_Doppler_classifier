"""
The Siamese embedding network.

It does not output class scores. It maps an image to a unit-length embedding
vector, and classification happens later by KNN over those vectors (see
classify.py). Two images of the same phase should land close together.

Design notes from tuning:

Embedding kept at 16 dims on purpose. 64 also worked, but results were
"too good" to be an interesting test of small-data behaviour.

AdaptiveAvgPool2d is the load-bearing layer. Averaging the feature map
over space is what lets the net separate Doppler phases 3 and 4; without
it, accuracy collapses to ~60% and those phases 3 and 4 in "doppler" 
and 3 and 4 in "gamma doppler" blur together.
"""

import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=cfg.EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),
            nn.Linear(8, embedding_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=1)


def build_model(device="cpu", embedding_dim=cfg.EMBEDDING_DIM):
    model = SiameseNet(embedding_dim).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"SiameseNet: {n:,} parameters, {embedding_dim}-dim embedding")
    return model
