"""Test 2 (spec §12): the attention-rate blend."""

import math

import torch
import torch.nn.functional as F

from clockpizza.model import Transformer


def _qkv(attn, x):
    v = torch.einsum("hkd,bpd->bhpk", attn.W_V, x)
    q = torch.einsum("hkd,bpd->bhpk", attn.W_Q, x)
    k = torch.einsum("hkd,bpd->bhpk", attn.W_K, x)
    return q, k, v


def _flatten_heads(z):
    b, _, n, _ = z.shape
    return z.permute(0, 2, 1, 3).reshape(b, n, -1)


def test_alpha_zero_is_the_sum_of_values():
    m = Transformer(attn_coeff=0.0, seed=0)
    attn = m.blocks[0].attn
    x = torch.randn(5, 2, 128)
    _, _, v = _qkv(attn, x)

    out = attn(x)
    # every query position gets the same output: the *sum* (not mean) over positions
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6)
    expected = _flatten_heads(v.sum(dim=2, keepdim=True).expand_as(v)) @ attn.W_O.T
    assert torch.allclose(out, expected, atol=1e-5)


def test_alpha_one_is_plain_softmax_attention():
    m = Transformer(attn_coeff=1.0, seed=0)
    attn = m.blocks[0].attn
    x = torch.randn(5, 2, 128)
    q, k, v = _qkv(attn, x)

    A = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(attn.d_head), dim=-1)
    assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-6)
    expected = _flatten_heads(A @ v) @ attn.W_O.T
    assert torch.allclose(attn(x), expected, atol=1e-5)


def test_blend_rows_sum_to_two_minus_alpha():
    """J is all-ones, not ones / n, so rows sum to 2 - alpha at n_ctx = 2."""
    for alpha in (0.0, 0.25, 1.0):
        m = Transformer(attn_coeff=alpha, seed=0)
        attn = m.blocks[0].attn
        x = torch.randn(3, 2, 128)
        q, k, _ = _qkv(attn, x)
        A = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(attn.d_head), dim=-1)
        A = A * alpha + (1 - alpha)
        assert torch.allclose(A.sum(-1), torch.full_like(A.sum(-1), 2 - alpha), atol=1e-6)


def test_mask_buffer_exists_but_is_not_applied():
    m = Transformer(attn_coeff=1.0, seed=0)
    attn = m.blocks[0].attn
    assert torch.equal(attn.mask, torch.tril(torch.ones(2, 2)))
    # bidirectional: position 0 attends to position 1, so swapping it changes the output
    x = torch.randn(1, 2, 128)
    x2 = x.clone()
    x2[:, 1] = torch.randn(128)
    assert not torch.allclose(attn(x)[:, 0], attn(x2)[:, 0], atol=1e-4)
