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


def shared_example_fraction(left: set[str], right: set[str]) -> float:
    """Jaccard overlap of two example sets, the ``e`` of the paper's Algorithm 1."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def apply_example_penalty(
    distance: np.ndarray,
    examples: list[set[str]],
    lower: float,
    upper: float,
    penalty_mode: str = "heuristic",
) -> np.ndarray:
    """Add ``(upper - lower) * fraction^2`` to pairs farther than ``lower``.

    ``penalty_mode`` selects which fraction is squared, and the two choices move
    merging in opposite directions:

    ``"heuristic"`` (default)
        The fraction of examples that do **not** overlap, so codes drawn from
        different excerpts are pushed apart and codes grounded in the same data
        merge more readily. This is what the clustering heuristic accompanying
        the paper does, and it is the behaviour this package has always had.

    ``"paper"``
        The paper's printed Algorithm 1, which squares the Jaccard *overlap*
        and so makes codes grounded in the same excerpts harder to merge.
        Available for reproducing the published algorithm exactly; it is not
        recommended, because it penalises the very evidence that two codes
        describe one concept.

    The choice is consequential rather than cosmetic: on a 1520-code space the
    two settings leave 539 and 886 codes merged respectively.
    """
    if penalty_mode not in ("heuristic", "paper"):
        raise ValueError("penalty_mode must be 'heuristic' or 'paper'")
    measure = (
        shared_example_fraction if penalty_mode == "paper" else unique_example_fraction
    )
    adjusted = np.array(distance, dtype=float, copy=True)
    penalty = upper - lower
    count = adjusted.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            if adjusted[i, j] <= lower:
                continue
            fraction = measure(examples[i], examples[j])
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
    penalty_mode: str = "heuristic",
) -> list[list[int]]:
    """Cut a dendrogram with the paper's lower and upper thresholds.

    A node merges when its linkage distance is at most the lower threshold.
    It never merges above the upper threshold. Between the two, the allowed
    distance shrinks as the merged example set grows past the average of the
    candidate nodes, so a large or sparse cluster does not absorb nearby but
    distinct codes.

    ``penalty_mode`` is passed to :func:`apply_example_penalty`; see it for what
    the two settings mean and how much they differ.
    """
    if lower > upper:
        raise ValueError("lower threshold must be <= upper threshold")
    count = distance.shape[0]
    if count == 0:
        return []
    if count == 1:
        return [[0]]

    adjusted = apply_example_penalty(
        distance, examples, lower, upper, penalty_mode=penalty_mode
    )
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

    # The paper's second penalty normalises a node's unique-example count by
    # ``count_max - count_avg``, both measured "across candidate nodes" -- the
    # internal nodes of the dendrogram, the only places a merge is decided.
    internal_counts: list[int] = []

    def collect(node) -> None:
        if node.is_leaf():
            return
        internal_counts.append(len(examples_of(node)))
        collect(node.get_left())
        collect(node.get_right())

    collect(root)
    average_size = float(np.mean(internal_counts)) if internal_counts else 1.0
    maximum_size = float(np.max(internal_counts)) if internal_counts else 1.0
    penalty_span = max(maximum_size - average_size, 1e-9)
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
