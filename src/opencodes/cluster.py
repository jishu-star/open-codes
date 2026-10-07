"""Hierarchical merges on cosine distance, including the two-threshold penalty cut."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import squareform

# Distances that differ only by floating-point noise should compare equal.
_ATOL = 1e-8


def unique_example_fraction(left: set[str], right: set[str]) -> float:
    """Share of the union that is not in both example sets.

    The paper's printed formula adds a penalty proportional to example overlap.
    That would push codes grounded in the same excerpts farther apart. The
    accompanying clustering heuristic does the opposite: it penalizes the
    fraction of examples that do *not* overlap, so codes about different data
    are harder to merge. This function implements that heuristic.
    """
    union = left | right
    if not union:
        return 0.0
    return len(left.symmetric_difference(right)) / len(union)


def apply_example_penalty(
    distance: np.ndarray,
    examples: list[set[str]],
    lower: float,
    upper: float,
) -> np.ndarray:
    """Add ``(upper - lower) * unique_fraction^2`` to pairs farther than ``lower``."""
    adjusted = np.array(distance, dtype=float, copy=True)
    penalty = upper - lower
    count = adjusted.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            if adjusted[i, j] <= lower:
                continue
            fraction = unique_example_fraction(examples[i], examples[j])
            bump = penalty * fraction * fraction
            adjusted[i, j] += bump
            adjusted[j, i] += bump
    return adjusted


def clusters_at_threshold(distance: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage clusters. Pairs whose distance is within ``threshold`` can merge."""
    count = distance.shape[0]
    if count == 0:
        return []
    if count == 1:
        return [[0]]
    condensed = squareform(np.maximum(distance, 0.0), checks=False)
    linked = linkage(condensed, method="average")
    labels = fcluster(linked, t=threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return list(groups.values())


def clusters_with_penalties(
    distance: np.ndarray,
    examples: list[set[str]],
    lower: float,
    upper: float,
) -> list[list[int]]:
    """Cut a dendrogram with the paper's lower and upper thresholds.

    A node merges when its linkage distance is at most the lower threshold.
    It never merges above the upper threshold. Between the two, the allowed
    distance shrinks as the merged example set grows past the average, so a
    large or sparse cluster does not absorb nearby but distinct codes.
    """
    if lower > upper:
        raise ValueError("lower threshold must be <= upper threshold")
    count = distance.shape[0]
    if count == 0:
        return []
    if count == 1:
        return [[0]]

    adjusted = apply_example_penalty(distance, examples, lower, upper)
    condensed = squareform(np.maximum(adjusted, 0.0), checks=False)
    linked = linkage(condensed, method="average")
    root = to_tree(linked)

    cached: dict[int, set[str]] = {}

    def examples_of(node) -> set[str]:
        if node.id in cached:
            return cached[node.id]
        if node.is_leaf():
            cached[node.id] = set(examples[node.id])
        else:
            cached[node.id] = examples_of(node.get_left()) | examples_of(node.get_right())
        return cached[node.id]

    examples_of(root)
    populated = [len(bucket) for bucket in examples if len(bucket) > 1]
    average_size = float(np.mean(populated)) if populated else 1.0
    # The reference heuristic treats 3x the average as the top of the penalty.
    penalty_span = max(average_size * 2.0, 1e-9)
    penalty = upper - lower

    accepted: list[list[int]] = []

    def leaves_of(node) -> list[int]:
        if node.is_leaf():
            return [int(node.id)]
        return leaves_of(node.get_left()) + leaves_of(node.get_right())

    def walk(node) -> None:
        if node.is_leaf():
            accepted.append([int(node.id)])
            return
        size = len(examples_of(node))
        overrun = min(1.0, max(0.0, (size - average_size) / penalty_span))
        criteria = max(upper - (overrun * overrun) * penalty, lower)
        if node.dist <= criteria + _ATOL:
            accepted.append(leaves_of(node))
            return
        walk(node.get_left())
        walk(node.get_right())

    walk(root)
    return accepted


def neighbor_indexes(distance: np.ndarray, threshold: float) -> list[list[int]]:
    """Codes within ``threshold`` that were not merged stay linked as neighbors."""
    count = distance.shape[0]
    neighbors: list[list[int]] = []
    for i in range(count):
        row = [
            j
            for j in range(count)
            if i != j and distance[i, j] <= threshold + _ATOL
        ]
        neighbors.append(row)
    return neighbors
