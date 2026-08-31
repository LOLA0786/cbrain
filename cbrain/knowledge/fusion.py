"""Deterministic Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> tuple[tuple[str, float], ...]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + (1.0 / (k + rank))
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def merge_score_maps(
    *maps: Mapping[str, float],
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for mapping in maps:
        for key, value in mapping.items():
            merged[key] = merged.get(key, 0.0) + value
    return merged


__all__ = ["RRF_K", "merge_score_maps", "reciprocal_rank_fusion"]
