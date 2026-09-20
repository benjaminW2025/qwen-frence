#!/usr/bin/env python3
"""Shared controls for exact-regime K-step CUDA graph decode experiments."""

from __future__ import annotations

import statistics
import time

import torch


UNROLL_STEPS = (2, 4, 8)


def stage_unrolled_case(torch, cfg, batch, context, max_steps, dtype, seed, device):
    """Allocate an empty fixed-layout cache and exact metadata for every step."""
    from paged_kv_cache import PagedKVCache

    block = 16
    pages = (context + max_steps + block - 1) // block
    cache = PagedKVCache(cfg, batch, batch * pages, block, device, dtype)
    cache.block_tables = [list(range(row * pages, (row + 1) * pages))
                          for row in range(batch)]
    cache.cur_lens = [context] * batch
    table = torch.arange(batch * pages, device=device, dtype=torch.int32).view(batch, pages)
    metadata = []
    for step in range(max_steps):
        position = context + step
        positions = torch.full((batch,), position, device=device, dtype=torch.int32)
        lengths = torch.full((batch,), position + 1, device=device, dtype=torch.int32)
        page_ids = table[:, position // block].to(torch.long)
        slots = page_ids * block + position % block
        metadata.append((positions, lengths, table, slots))
    generator = torch.Generator(device=device).manual_seed(seed)
    first_ids = torch.randint(1, cfg.vocab, (batch, 1), device=device,
                              generator=generator)
    return cache, tuple(metadata), first_ids, generator


def initialize_cache(torch, cache, generator):
    """Fill staged K/V only after graph capture warmups have finished."""
    for tensor in (*cache.k_pool, *cache.v_pool):
        tensor.normal_(generator=generator)
    torch.cuda.synchronize(cache.k_pool[0].device)


def future_slots(metadata):
    return torch.stack([row[3] for row in metadata], dim=0)


def snapshot_slots(cache, slots):
    """Copy only K future positions instead of duplicating the multi-GiB cache."""
    flat_slots = slots.reshape(-1)
    return tuple(tensor.view(-1, tensor.shape[2], tensor.shape[3]).index_select(
        0, flat_slots).clone() for tensor in (*cache.k_pool, *cache.v_pool))


def restore_slots(cache, slots, snapshot):
    flat_slots = slots.reshape(-1)
    tensors = (*cache.k_pool, *cache.v_pool)
    if len(snapshot) != len(tensors):
        raise ValueError("KV snapshot does not match the cache")
    for tensor, values in zip(tensors, snapshot):
        tensor.view(-1, tensor.shape[2], tensor.shape[3]).index_copy_(
            0, flat_slots, values)


def eager_trajectory(model, cache, first_ids, metadata, *, attention_policy="splitk"):
    """Authoritative repeated-forward trajectory retaining every logit tensor."""
    from paged_graph_decoder import graph_decode_forward

    token = first_ids
    logits = []
    tokens = []
    max_context = int(metadata[-1][1][0])
    for positions, lengths, table, slots in metadata:
        output = graph_decode_forward(
            model, cache, token, positions, lengths, table, slots,
            decode_attention_policy=attention_policy,
            max_decode_context_length=max_context,
        )
        logits.append(output)
        token = output.argmax(-1).reshape(first_ids.shape)
        tokens.append(token.reshape(-1))
    return torch.stack(tokens), tuple(logits)


def trajectory_accuracy(torch, candidate_tokens, candidate_logits,
                        reference_tokens, reference_logits, atol):
    """Comprehensive per-step logit and token comparison without early abort."""
    if len(candidate_logits) != len(reference_logits):
        raise ValueError("candidate and reference logit trajectories differ in length")
    rows = []
    outside_total = 0
    compared_total = 0
    max_error = 0.0
    mean_error_numerator = 0.0
    first_token_divergence = None
    for step, (candidate, reference) in enumerate(zip(candidate_logits, reference_logits)):
        difference = (candidate.float() - reference.float()).abs()
        outside = int((difference > atol).sum())
        count = difference.numel()
        candidate_ids = candidate.argmax(-1).reshape(-1)
        reference_ids = reference.argmax(-1).reshape(-1)
        token_matches = int((candidate_ids == reference_ids).sum())
        if first_token_divergence is None and token_matches != reference_ids.numel():
            first_token_divergence = step
        step_max = float(difference.max())
        step_mean = float(difference.mean())
        rows.append({
            "step": step, "max_absolute_error": step_max,
            "mean_absolute_error": step_mean,
            "logits_outside_atol": outside, "logits_compared": count,
            "matching_tokens": token_matches, "total_tokens": reference_ids.numel(),
        })
        outside_total += outside
        compared_total += count
        max_error = max(max_error, step_max)
        mean_error_numerator += step_mean * count
    trajectory_matches = int((candidate_tokens == reference_tokens).sum())
    trajectory_total = reference_tokens.numel()
    return {
        "status": ("pass" if outside_total == 0
                   and trajectory_matches == trajectory_total else "fail"),
        "steps": rows, "max_absolute_error": max_error,
        "mean_absolute_error": mean_error_numerator / compared_total,
        "logits_outside_atol": outside_total, "logits_compared": compared_total,
        "matching_tokens": trajectory_matches, "total_tokens": trajectory_total,
        "first_token_divergence_step": first_token_divergence,
    }


def kv_accuracy(torch, candidate, reference, atol):
    if len(candidate) != len(reference):
        raise ValueError("KV snapshots differ in length")
    maximum = mean_sum = elements = outside = 0
    for actual, expected in zip(candidate, reference):
        difference = (actual.float() - expected.float()).abs()
        maximum = max(maximum, float(difference.max()))
        mean_sum += float(difference.sum())
        elements += difference.numel()
        outside += int((difference > atol).sum())
    return {"status": "pass" if outside == 0 else "fail",
            "max_absolute_error": maximum,
            "mean_absolute_error": mean_sum / elements,
            "values_outside_atol": outside, "values_compared": elements}


def timed_cuda_wall(torch, operation, repetitions, warmups=1):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    gpu_samples, wall_samples = [], []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        operation()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000)
        gpu_samples.append(float(start.elapsed_time(end)))
    return {"gpu_median_ms": statistics.median(gpu_samples),
            "wall_median_ms": statistics.median(wall_samples),
            "gpu_samples_ms": gpu_samples, "wall_samples_ms": wall_samples}
