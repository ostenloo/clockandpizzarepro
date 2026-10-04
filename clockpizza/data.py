"""Task and data: (a + b) mod p over all ordered pairs (spec §5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

P = 59
TRAIN_FRAC = 0.8


@dataclass(frozen=True)
class Dataset:
    """All p^2 ordered pairs, enumerated row-major: index i -> (a, b) = (i // p, i % p)."""

    inputs: torch.Tensor  # (p*p, n_ctx) long
    labels: torch.Tensor  # (p*p,) long
    a: torch.Tensor  # (p*p,) long
    b: torch.Tensor  # (p*p,) long
    p: int
    vocab_size: int
    n_ctx: int

    def __len__(self) -> int:
        return self.inputs.shape[0]


def make_dataset(p: int = P, diff_vocab: bool = False, eqn_sign: bool = False) -> Dataset:
    """Build the full dataset.

    Base: tokens 0..p-1, sequence [a, b], target (a + b) % p.
    ``diff_vocab``: second token is shifted to b + p, vocabulary 2p.
    ``eqn_sign``: append an '=' token whose id is the last vocabulary index.
    """
    idx = torch.arange(p * p, dtype=torch.long)
    a = idx // p
    b = idx % p
    labels = (a + b) % p

    vocab_size = 2 * p if diff_vocab else p
    cols = [a, b + p if diff_vocab else b]
    if eqn_sign:
        vocab_size += 1
        cols.append(torch.full_like(a, vocab_size - 1))

    inputs = torch.stack(cols, dim=1)
    return Dataset(
        inputs=inputs,
        labels=labels,
        a=a,
        b=b,
        p=p,
        vocab_size=vocab_size,
        n_ctx=inputs.shape[1],
    )


def split_indices(
    seed: int, n: int = P * P, frac: float = TRAIN_FRAC
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute the n indices from the run seed; the first int(frac * n) are train.

    Returns (train_idx, val_idx) as long tensors. For p = 59 this is 2784 / 697.
    """
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)
    train_size = int(frac * n)
    return perm[:train_size], perm[train_size:]


def split_masks(seed: int, n: int = P * P, frac: float = TRAIN_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    """Same split as :func:`split_indices`, returned as boolean masks of length n."""
    train_idx, val_idx = split_indices(seed, n=n, frac=frac)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    return train_mask, ~train_mask
