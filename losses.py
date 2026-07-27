"""Triplet loss with hard mining — the training signal for the embedding.

Instead of a standard classification loss, the network is trained so that, for
each image (the anchor), the hardest same-phase image is pulled closer than
the hardest different-phase image, by at least a margin. Hardest = the
same-class pair that is currently furthest apart, and the different-class pair
that is currently closest. Focusing on the hard cases is what teaches the net
to separate near-identical phases.
"""

import torch


def pairwise_distances(embeddings):
    """
    Euclidean distance between every pair of embeddings in a batch.
    """
    dot = embeddings @ embeddings.T
    sq = torch.diagonal(dot)
    d = sq.unsqueeze(1) - 2.0 * dot + sq.unsqueeze(0)
    d = torch.clamp(d, min=0.0)

    mask = (d == 0).float()
    d = torch.sqrt(d + mask * 1e-16) * (1.0 - mask)
    return d


def triplet_hard_loss(embeddings, labels, margin=1.0):
    """
    Batch-hard triplet loss (mean over anchors).
    """
    pdist = pairwise_distances(embeddings)
    B = labels.size(0)

    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same.float() - torch.eye(B, device=embeddings.device)   # same class
    neg_mask = 1.0 - same.float()                                       # different class

    # hardest positive - furthest same-class image
    hardest_pos = (pdist * pos_mask).max(dim=1).values
    # hardest negative - closest different-class image
    masked_neg = pdist + pdist.max() * (1.0 - neg_mask)
    hardest_neg = masked_neg.min(dim=1).values

    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return loss.mean()
