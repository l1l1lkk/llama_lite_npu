"""Shared local and vocabulary-parallel token sampling.

Qwen3 tensor parallelism shards the LM head along the vocabulary dimension.
Greedy decoding therefore needs only one maximum value/token pair per rank.
Top-P decoding first exchanges bounded candidates. It falls back to full-logit
gathering when the candidate set cannot prove that it contains the exact
nucleus.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import torch


NumberOrSequence = Union[float, Sequence[float], torch.Tensor]


class SamplingStats:
    """Low-cardinality counters for distributed sampling decisions."""

    def __init__(
        self,
        candidate_rows: int = 0,
        fallback_rows: int = 0,
        full_logit_gather_batches: int = 0,
    ) -> None:
        self.candidate_rows = int(candidate_rows)
        self.fallback_rows = int(fallback_rows)
        self.full_logit_gather_batches = int(full_logit_gather_batches)


class SamplingResult:
    """Device sampled tokens plus optional sampling-path stats."""

    def __init__(self, tokens: torch.Tensor, stats: SamplingStats) -> None:
        self.tokens = tokens
        self.stats = stats


def format_top_p_setting(temperature: float, top_p: float) -> str:
    """Describe whether Top-P participates in the selected sampling mode."""
    if float(temperature) <= 0:
        return "inactive (temperature=0)"
    return str(float(top_p))


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits[:, -1, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(
        f"logits must have shape [batch, vocab] or [batch, seq, vocab], "
        f"got {tuple(logits.shape)}"
    )


def _parameter_values(
    value: NumberOrSequence,
    batch_size: int,
) -> list[float]:
    if isinstance(value, torch.Tensor):
        values = value.detach().reshape(-1).cpu().tolist()
    elif isinstance(value, Sequence):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    if len(values) == 1 and batch_size != 1:
        values *= batch_size
    if len(values) != batch_size:
        raise ValueError(
            f"sampling parameter has {len(values)} values for "
            f"batch_size={batch_size}"
        )
    return values


def _sample_top_p_row(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, descending=True
    )
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities > top_p
    sorted_probabilities.masked_fill_(remove, 0.0)
    sorted_probabilities.div_(sorted_probabilities.sum())
    sampled_position = torch.multinomial(
        sorted_probabilities, num_samples=1
    )
    return sorted_indices.gather(0, sampled_position).squeeze(0)


def sample_local_logits(
    logits: torch.Tensor,
    temperature: NumberOrSequence = 0.0,
    top_p: NumberOrSequence = 1.0,
) -> torch.Tensor:
    """Sample full-vocabulary logits without distributed communication."""
    local_logits = _last_token_logits(logits)
    batch_size = local_logits.shape[0]
    temperatures = _parameter_values(temperature, batch_size)
    top_ps = _parameter_values(top_p, batch_size)

    tokens = []
    for row in range(batch_size):
        row_temperature = temperatures[row]
        if row_temperature <= 0:
            tokens.append(torch.argmax(local_logits[row], dim=-1))
        else:
            tokens.append(
                _sample_top_p_row(
                    local_logits[row],
                    row_temperature,
                    top_ps[row],
                )
            )
    return torch.stack(tokens).to(dtype=torch.long)


def select_greedy_from_shards(
    gathered_values: torch.Tensor,
    gathered_global_ids: torch.Tensor,
) -> torch.Tensor:
    """Select global token IDs from ``[world_size, batch]`` rank maxima."""
    if gathered_values.shape != gathered_global_ids.shape:
        raise ValueError("gathered values and ids must have identical shapes")
    winning_ranks = torch.argmax(gathered_values, dim=0, keepdim=True)
    return torch.gather(
        gathered_global_ids, dim=0, index=winning_ranks
    ).squeeze(0).to(dtype=torch.long)


def candidate_nucleus_is_complete(
    *,
    sorted_logits: torch.Tensor,
    exact_probabilities: torch.Tensor,
    top_p: float,
    max_excluded_logit: torch.Tensor,
) -> bool:
    """Return whether candidates provably contain the exact Top-P nucleus.

    Candidate probabilities must use the exact global softmax denominator. The
    nucleus is complete only when its cumulative mass reaches ``top_p`` and the
    final included logit is not lower than any excluded candidate.
    """
    if sorted_logits.numel() == 0:
        return False
    cumulative = torch.cumsum(exact_probabilities, dim=-1)
    included = cumulative - exact_probabilities <= top_p
    included_count = int(included.sum().item())
    if included_count == 0:
        included_count = 1
    if float(cumulative[-1]) < float(top_p):
        return False
    cutoff_logit = sorted_logits[included_count - 1]
    return bool((cutoff_logit >= max_excluded_logit).item())


def _tp_state():
    from .executor.tp_utils import (
        get_tp_config,
        get_tp_group,
        tp_is_initialized,
    )

    config = get_tp_config()
    return config, get_tp_group(), tp_is_initialized()


def _all_gather_stack(tensor: torch.Tensor, world_size: int, group):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(chunks, tensor, group=group)
    return torch.stack(chunks, dim=0)


def _all_reduce_clone(tensor: torch.Tensor, op, group):
    result = tensor.clone()
    torch.distributed.all_reduce(result, op=op, group=group)
    return result


def gather_vocab_parallel_logits(local_logits: torch.Tensor) -> torch.Tensor:
    """Gather vocabulary shards only for APIs that require complete logits."""
    config, group, distributed = _tp_state()
    if not distributed:
        return local_logits
    chunks = [
        torch.empty_like(local_logits) for _ in range(config.world_size)
    ]
    torch.distributed.all_gather(chunks, local_logits, group=group)
    return torch.cat(chunks, dim=-1)


def _sample_vocab_parallel_greedy(
    local_logits: torch.Tensor,
    *,
    config,
    group,
) -> torch.Tensor:
    """Select an entire Greedy batch with one small candidate collective."""
    local_values, local_ids = torch.max(local_logits, dim=-1)
    global_ids = local_ids.to(torch.long) + (
        int(config.rank) * local_logits.shape[-1]
    )
    if (int(config.rank) + 1) * local_logits.shape[-1] > 2**24:
        raise RuntimeError(
            "vocabulary-parallel token IDs exceed exact float32 range"
        )

    packed_candidates = torch.stack(
        [local_values.float(), global_ids.float()],
        dim=-1,
    )
    gathered = _all_gather_stack(
        packed_candidates,
        config.world_size,
        group,
    )
    return select_greedy_from_shards(
        gathered[..., 0],
        gathered[..., 1].to(torch.long),
    )



def _sample_vocab_parallel_top_p_batch(
    local_logits: torch.Tensor,
    *,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    candidate_k: int,
    config,
    group,
    return_stats: bool = False,
):
    """Sample a Top-P batch with batched candidate exchange.

    The exact path exchanges only the top ``candidate_k`` logits from each
    vocabulary shard.  If those candidates cannot prove that they contain the
    exact global nucleus for any row, all ranks perform one full-logit gather
    for the batch and rank 0 samples only the fallback rows.
    """

    batch_size, local_vocab_size = local_logits.shape
    device = local_logits.device
    vocab_offset = int(config.rank) * local_vocab_size
    temperature_tensor = torch.tensor(
        [float(value) for value in temperatures],
        dtype=torch.float32,
        device=device,
    ).reshape(batch_size, 1)
    scaled = local_logits.float() / temperature_tensor

    global_max = _all_reduce_clone(
        torch.max(scaled, dim=-1).values,
        torch.distributed.ReduceOp.MAX,
        group,
    )
    local_exp_sum = torch.exp(scaled - global_max.reshape(-1, 1)).sum(dim=-1)
    global_exp_sum = _all_reduce_clone(
        local_exp_sum,
        torch.distributed.ReduceOp.SUM,
        group,
    )

    k = min(max(1, int(candidate_k)), local_vocab_size)
    top_count = min(k + 1, local_vocab_size)
    local_top_logits, local_top_ids = torch.topk(scaled, top_count, dim=-1)
    if top_count > k:
        max_excluded = local_top_logits[:, k]
        local_top_logits = local_top_logits[:, :k]
        local_top_ids = local_top_ids[:, :k]
    else:
        max_excluded = torch.full(
            (batch_size,),
            float("-inf"),
            device=device,
            dtype=scaled.dtype,
        )

    gathered_logits = _all_gather_stack(
        local_top_logits, int(config.world_size), group
    )
    gathered_ids = _all_gather_stack(
        local_top_ids.to(torch.long) + vocab_offset,
        int(config.world_size),
        group,
    )
    gathered_excluded = _all_gather_stack(
        max_excluded, int(config.world_size), group
    )

    tokens = torch.zeros((batch_size,), dtype=torch.long, device=device)
    fallback_rows: list[int] = []
    row_candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for row in range(batch_size):
        row_logits = gathered_logits[:, row, :].reshape(-1)
        row_ids = gathered_ids[:, row, :].reshape(-1)
        sorted_logits, order = torch.sort(row_logits, descending=True)
        sorted_ids = row_ids[order]
        exact_probabilities = torch.exp(
            sorted_logits - global_max[row]
        ) / global_exp_sum[row]
        complete = candidate_nucleus_is_complete(
            sorted_logits=sorted_logits,
            exact_probabilities=exact_probabilities,
            top_p=float(top_ps[row]),
            max_excluded_logit=torch.max(gathered_excluded[:, row]),
        )
        row_candidates.append((sorted_logits, sorted_ids, exact_probabilities))
        if not complete:
            fallback_rows.append(row)

    full_logits = None
    full_logit_gather_batches = 0
    if fallback_rows:
        full_logits = torch.cat(
            list(_all_gather_stack(local_logits, int(config.world_size), group)),
            dim=-1,
        )
        full_logit_gather_batches = 1

    if int(config.rank) == 0:
        for row in range(batch_size):
            if row in fallback_rows:
                tokens[row] = _sample_top_p_row(
                    full_logits[row],
                    float(temperatures[row]),
                    float(top_ps[row]),
                ).to(torch.long)
                continue
            _, sorted_ids, exact_probabilities = row_candidates[row]
            cumulative = torch.cumsum(exact_probabilities, dim=-1)
            remove = cumulative - exact_probabilities > float(top_ps[row])
            candidate_probabilities = exact_probabilities.masked_fill(remove, 0.0)
            candidate_probabilities = (
                candidate_probabilities / candidate_probabilities.sum()
            )
            sampled_position = torch.multinomial(
                candidate_probabilities, num_samples=1
            )
            tokens[row] = sorted_ids[sampled_position].reshape(()).to(torch.long)

    torch.distributed.broadcast(tokens, src=0, group=group)
    stats = SamplingStats(
        candidate_rows=batch_size,
        fallback_rows=len(fallback_rows),
        full_logit_gather_batches=full_logit_gather_batches,
    )
    if return_stats:
        return tokens, stats
    return tokens


def _sample_vocab_parallel_row(
    local_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    candidate_k: int,
    config,
    group,
) -> torch.Tensor:
    local_vocab_size = local_logits.shape[-1]
    vocab_offset = config.rank * local_vocab_size

    if temperature <= 0:
        local_value, local_id = torch.max(local_logits, dim=-1)
        global_id = local_id.to(torch.long) + vocab_offset
        gathered_values = _all_gather_stack(
            local_value.reshape(1), config.world_size, group
        )
        gathered_ids = _all_gather_stack(
            global_id.reshape(1), config.world_size, group
        )
        return select_greedy_from_shards(
            gathered_values, gathered_ids
        ).reshape(())

    scaled = local_logits.float() / float(temperature)
    global_max = _all_reduce_clone(
        torch.max(scaled),
        torch.distributed.ReduceOp.MAX,
        group,
    )
    local_exp_sum = torch.exp(scaled - global_max).sum()
    global_exp_sum = _all_reduce_clone(
        local_exp_sum,
        torch.distributed.ReduceOp.SUM,
        group,
    )

    k = min(max(1, int(candidate_k)), local_vocab_size)
    top_count = min(k + 1, local_vocab_size)
    local_top_logits, local_top_ids = torch.topk(scaled, top_count)
    if top_count > k:
        max_excluded = local_top_logits[k]
        local_top_logits = local_top_logits[:k]
        local_top_ids = local_top_ids[:k]
    else:
        max_excluded = torch.tensor(
            float("-inf"), device=scaled.device, dtype=scaled.dtype
        )

    gathered_logits = _all_gather_stack(
        local_top_logits, config.world_size, group
    ).reshape(-1)
    gathered_ids = _all_gather_stack(
        local_top_ids.to(torch.long) + vocab_offset,
        config.world_size,
        group,
    ).reshape(-1)
    gathered_excluded = _all_gather_stack(
        max_excluded.reshape(1), config.world_size, group
    )

    sorted_logits, order = torch.sort(gathered_logits, descending=True)
    sorted_ids = gathered_ids[order]
    exact_probabilities = torch.exp(
        sorted_logits - global_max
    ) / global_exp_sum
    complete = candidate_nucleus_is_complete(
        sorted_logits=sorted_logits,
        exact_probabilities=exact_probabilities,
        top_p=top_p,
        max_excluded_logit=torch.max(gathered_excluded),
    )

    if not complete:
        full_logits = gather_vocab_parallel_logits(
            local_logits.reshape(1, -1)
        ).reshape(-1)
        if config.rank == 0:
            token = _sample_top_p_row(
                full_logits, temperature, top_p
            ).to(torch.long)
        else:
            token = torch.zeros(
                (), dtype=torch.long, device=local_logits.device
            )
    elif config.rank == 0:
        cumulative = torch.cumsum(exact_probabilities, dim=-1)
        remove = cumulative - exact_probabilities > top_p
        candidate_probabilities = exact_probabilities.masked_fill(remove, 0.0)
        candidate_probabilities = (
            candidate_probabilities / candidate_probabilities.sum()
        )
        sampled_position = torch.multinomial(
            candidate_probabilities, num_samples=1
        )
        token = sorted_ids[sampled_position].reshape(()).to(torch.long)
    else:
        token = torch.zeros(
            (), dtype=torch.long, device=local_logits.device
        )

    torch.distributed.broadcast(token, src=0, group=group)
    return token


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: NumberOrSequence = 0.0,
    top_p: NumberOrSequence = 1.0,
    vocab_parallel: bool = False,
    candidate_k: int = 2048,
    return_stats: bool = False,
):
    """Sample next-token IDs, preserving a device tensor result."""
    local_logits = _last_token_logits(logits)
    if not vocab_parallel:
        sampled = sample_local_logits(local_logits, temperature, top_p)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.broadcast(sampled, src=0)
        if return_stats:
            return SamplingResult(sampled, SamplingStats())
        return sampled

    config, group, distributed = _tp_state()
    if not distributed:
        sampled = sample_local_logits(local_logits, temperature, top_p)
        if return_stats:
            return SamplingResult(sampled, SamplingStats())
        return sampled

    batch_size = local_logits.shape[0]
    temperatures = _parameter_values(temperature, batch_size)
    top_ps = _parameter_values(top_p, batch_size)
    if all(row_temperature <= 0 for row_temperature in temperatures):
        sampled = _sample_vocab_parallel_greedy(
            local_logits,
            config=config,
            group=group,
        )
        if return_stats:
            return SamplingResult(sampled, SamplingStats())
        return sampled

    if any(row_temperature <= 0 for row_temperature in temperatures):
        # Mixed greedy/sampling rows are uncommon and are kept on the older
        # row-wise path for correctness. Homogeneous Top-P batches use the
        # batched candidate exchange below.
        sampled = [
            _sample_vocab_parallel_row(
                local_logits[row],
                temperature=temperatures[row],
                top_p=top_ps[row],
                candidate_k=candidate_k,
                config=config,
                group=group,
            )
            for row in range(batch_size)
        ]
        tokens = torch.stack(sampled).reshape(-1)
        if return_stats:
            return SamplingResult(tokens, SamplingStats(candidate_rows=batch_size))
        return tokens

    tokens, stats = _sample_vocab_parallel_top_p_batch(
        local_logits,
        temperatures=temperatures,
        top_ps=top_ps,
        candidate_k=candidate_k,
        config=config,
        group=group,
        return_stats=True,
    )
    if return_stats:
        return SamplingResult(tokens, stats)
    return tokens
