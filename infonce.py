import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):


    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys, temperature=self.temperature, reduction=self.reduction)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean'):
    # Inputs all have 2 dimensions.
    if query.dim() != 2 or positive_key.dim() != 2 or (negative_keys is not None and negative_keys.dim() != 2):
        raise ValueError('query, positive_key and negative_keys should all have 2 dimensions.')

    # Each query sample is paired with exactly one positive key sample.
    if len(query) != len(positive_key):
        raise ValueError('query and positive_key must have the same number of samples.')

    # Embedding vectors should have same number of components.
    if query.shape[1] != positive_key.shape[1] != (positive_key.shape[1] if negative_keys is None else negative_keys.shape[1]):
        raise ValueError('query, positive_key and negative_keys should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        # Cosine between all query-negative combinations
        negative_logits = query @ transpose(negative_keys)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction), positive_logit.mean(), negative_logits.mean()


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]