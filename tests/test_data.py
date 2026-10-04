"""Test 1 (spec §12): data and splits."""

import torch

from clockpizza.data import P, make_dataset, split_indices, split_masks


def test_all_pairs_and_labels():
    d = make_dataset()
    assert len(d) == P * P == 3481
    assert d.vocab_size == 59 and d.n_ctx == 2
    # row-major enumeration: index i -> (a, b) = (i // p, i % p)
    for i in (0, 1, 58, 59, 60, 3480):
        a, b = i // P, i % P
        assert d.inputs[i].tolist() == [a, b]
        assert d.labels[i].item() == (a + b) % P
    assert torch.equal(d.labels, (d.a + d.b) % P)


def test_split_sizes_disjoint_and_covering():
    train, val = split_indices(0)
    assert len(train) == 2784 and len(val) == 697
    both = torch.cat([train, val]).sort().values
    assert torch.equal(both, torch.arange(P * P))  # disjoint and covering


def test_split_is_seed_determined():
    a1, b1 = split_indices(7)
    a2, b2 = split_indices(7)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    a3, _ = split_indices(8)
    assert not torch.equal(a1, a3)


def test_split_masks_agree_with_indices():
    train, val = split_indices(3)
    tm, vm = split_masks(3)
    assert tm.sum().item() == 2784 and vm.sum().item() == 697
    assert torch.equal(tm.nonzero().ravel().sort().values, train.sort().values)
    assert torch.equal(vm.nonzero().ravel().sort().values, val.sort().values)


def test_variants():
    d = make_dataset(diff_vocab=True)
    assert d.vocab_size == 118 and d.n_ctx == 2
    assert d.inputs[60].tolist() == [1, 1 + P]

    d = make_dataset(eqn_sign=True)
    assert d.vocab_size == 60 and d.n_ctx == 3
    assert d.inputs[60].tolist() == [1, 1, 59]

    d = make_dataset(diff_vocab=True, eqn_sign=True)
    assert d.vocab_size == 119 and d.n_ctx == 3
    assert d.inputs[60].tolist() == [1, 1 + P, 118]
    assert torch.equal(d.labels, (d.a + d.b) % P)
