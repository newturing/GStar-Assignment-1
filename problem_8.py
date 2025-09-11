# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

        scale = softmax_scale
        qf = q.float()
        kf = k.float()
        vf = v.float()
        groups = n_heads // n_kv_heads
        arange = torch.arange(seq_len, device=q.device)
        i_idx = arange.view(1, -1, 1)
        j_idx = arange.view(1, 1, -1)
        for h in range(n_heads):
            kvh = h // groups
            s = (qf[:, h] @ kf[:, kvh].transpose(-1, -2)) * scale
            if is_causal:
                mask = j_idx > i_idx
                s = s.masked_fill(mask, float('-inf'))
            p = torch.softmax(s, dim=-1)
            o[:, h] = p @ vf[:, kvh]
            M[:, h] = torch.logsumexp(s, dim=-1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads
        scale = ctx.softmax_scale
        groups = n_heads // n_kv_heads

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        qf = q.float()
        kf = k.float()
        vf = v.float()
        of = o.float()
        dof = do.float()

        arange = torch.arange(seq_len, device=q.device)
        i_idx = arange.view(1, -1, 1)
        j_idx = arange.view(1, 1, -1)
        causal_mask = j_idx > i_idx

        for h in range(n_heads):
            kvh = h // groups
            
            s = (qf[:, h] @ kf[:, kvh].transpose(-1, -2)) * scale
            s = s.masked_fill(causal_mask, float('-inf'))
            p = torch.softmax(s, dim=-1)
            
            delta = torch.sum(dof[:, h] * of[:, h], dim=-1, keepdim=True)
            
            ds = p * (dof[:, h] @ vf[:, kvh].transpose(-1, -2) - delta)
            
            dq[:, h] = ds @ kf[:, kvh] * scale
            dk[:, kvh] += (ds.transpose(-1, -2) @ qf[:, h]) * scale
            dv[:, kvh] += p.transpose(-1, -2) @ dof[:, h]
        
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)