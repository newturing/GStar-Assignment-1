import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks
    
    q_start_pos = q_block_idx * BLOCK_M
    window_start = tl.maximum(SINK_SIZE, q_start_pos - WINDOW_SIZE)
    window_start = (window_start // BLOCK_N) * BLOCK_N

    # Phase 0: Sink blocks
    for start_n in range(0, SINK_SIZE, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        p_ij = tl.exp2(s_ij - m_new[:, None])
        
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        
        m_i = m_new

    # Phase 1: Window blocks
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        row_indices = q_offsets[:, None]
        col_indices = k_offsets[None, :]
        window_mask = (row_indices - col_indices) >= WINDOW_SIZE
        sink_mask = col_indices >= SINK_SIZE
        s_ij = tl.where(window_mask & sink_mask, float('-inf'), s_ij)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        p_ij = tl.exp2(s_ij - m_new[:, None])
        
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        
        m_i = m_new

    # Phase 2: Diagonal blocks
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        row_indices = q_offsets[:, None]
        col_indices = k_offsets[None, :]
        causal_mask = row_indices < col_indices
        window_mask = (row_indices - col_indices) >= WINDOW_SIZE
        sink_mask = col_indices >= SINK_SIZE
        s_ij = tl.where(causal_mask | (window_mask & sink_mask), float('-inf'), s_ij)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        p_ij = tl.exp2(s_ij - m_new[:, None])
        
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    PyTorch implementation of GQA + Sliding Window + Attention Sinks.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    orig_dtype = q.dtype
    scale = 1.0 / math.sqrt(head_dim)
    
    # Handle GQA: repeat K/V heads if needed
    if n_q_heads != n_kv_heads:
        num_groups = n_q_heads // n_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Create masks
    device = scores.device
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    # Sliding window mask: allow attention to tokens within window_size distance
    row_idx = torch.arange(seq_len, device=device)[:, None]
    col_idx = torch.arange(seq_len, device=device)[None, :]
    window_mask = (row_idx - col_idx) >= window_size
    
    # Sink mask: always allow attention to first sink_size tokens
    sink_mask = col_idx < sink_size
    
    # Combined mask: causal AND (outside window AND not in sink)
    combined_mask = causal_mask | (window_mask & ~sink_mask)
    
    # Apply mask
    scores = scores.masked_fill(combined_mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output.to(orig_dtype)