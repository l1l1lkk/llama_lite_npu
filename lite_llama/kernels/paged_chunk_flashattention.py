import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd


@triton.jit
def _paged_chunk_flash_attention_kernel(
    Q,
    KCache,
    VCache,
    O,
    ReqTokenTable,
    ReqIds,
    QStartLoc,
    ContextLen,
    QSeqLen,
    sm_scale,
    heads,
    num_kv_groups,
    stride_q_bs,
    stride_q_heads,
    stride_q_dim,
    stride_k_bs,
    stride_k_heads,
    stride_k_dim,
    stride_v_bs,
    stride_v_heads,
    stride_v_dim,
    stride_o_bs,
    stride_o_heads,
    stride_o_dim,
    stride_req_to_token_b,
    stride_req_to_token_s,
    HEAD_DIM: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch_idx = cur_bh // heads
    cur_head_idx = cur_bh % heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups

    req_id = tl.load(ReqIds + cur_batch_idx)
    q_start = tl.load(QStartLoc + cur_batch_idx)
    context_len = tl.load(ContextLen + cur_batch_idx)
    q_len = tl.load(QSeqLen + cur_batch_idx)

    block_start = block_m_idx * BLOCK_M_SIZE
    offs_m = block_start + tl.arange(0, BLOCK_M_SIZE)
    offs_n = tl.arange(0, BLOCK_N_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)

    q_logical_pos = context_len + offs_m
    q_offs = (
        (q_start + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < q_len, other=0.0)

    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, HEAD_DIM), dtype=tl.float32)

    block_end = tl.minimum(block_start + BLOCK_M_SIZE, q_len)
    kv_end = context_len + block_end
    for start_n in range(0, kv_end, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)
        key_pos = start_n + offs_n
        physical_pos = tl.load(
            ReqTokenTable
            + req_id * stride_req_to_token_b
            + key_pos * stride_req_to_token_s,
            mask=key_pos < kv_end,
            other=0,
        )

        k_offs = (
            physical_pos[None, :] * stride_k_bs
            + cur_kv_head_idx * stride_k_heads
            + offs_d[:, None] * stride_k_dim
        )
        k = tl.load(
            KCache + k_offs,
            mask=(key_pos[None, :] < kv_end),
            other=0.0,
        )

        qk = tl.dot(q, k)
        causal = q_logical_pos[:, None] >= key_pos[None, :]
        valid = (offs_m[:, None] < q_len) & (key_pos[None, :] < kv_end) & causal
        qk = tl.where(valid, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        d_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij
        acc = acc * alpha[:, None]

        v_offs = (
            physical_pos[:, None] * stride_v_bs
            + cur_kv_head_idx * stride_v_heads
            + offs_d[None, :] * stride_v_dim
        )
        v = tl.load(
            VCache + v_offs,
            mask=key_pos[:, None] < kv_end,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / d_i[:, None]
    o_offs = (
        (q_start + offs_m[:, None]) * stride_o_bs
        + cur_head_idx * stride_o_heads
        + offs_d[None, :] * stride_o_dim
    )
    tl.store(O + o_offs, acc, mask=offs_m[:, None] < q_len)


@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def paged_chunk_flash_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sm_scale,
    req_token_table: torch.Tensor,
    req_ids: torch.Tensor,
    q_start_loc: torch.Tensor,
    context_lens: torch.Tensor,
    q_seq_lens: torch.Tensor,
    max_q_len: int,
):
    output = torch.empty_like(q)
    if q.numel() == 0:
        return output

    n_heads, head_dim = q.shape[1], q.shape[2]
    batch_size = q_seq_lens.shape[0]
    block_size = 64
    num_kv_groups = q.shape[1] // k_cache.shape[1]
    grid = (triton.cdiv(int(max_q_len), block_size), batch_size * n_heads, 1)

    _paged_chunk_flash_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        req_token_table,
        req_ids,
        q_start_loc,
        context_lens,
        q_seq_lens,
        sm_scale,
        n_heads,
        num_kv_groups,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        req_token_table.stride(0),
        req_token_table.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_M_SIZE=block_size,
        BLOCK_N_SIZE=block_size,
    )
    return output
