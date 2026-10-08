"""Coverage, Overlap, Novelty, and Divergence against an aggregated code space."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from opencodes.cluster import neighbor_indexes
from opencodes.embed import cosine_distance_matrix
from opencodes.types import Code, Codebook


@dataclass(frozen=True)
class CoderMetrics:
    coder: str
    n_codes: int
    weight: float
    coverage: float
    overlap: float
    novelty: float
    divergence: float
    consolidated: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "coder": self.coder,
            "n_codes": self.n_codes,
            "consolidated": self.consolidated,
            "weight": self.weight,
            "coverage": self.coverage,
            "overlap": self.overlap,
            "novelty": self.novelty,
            "divergence": self.divergence,
        }


def evaluate_codebooks(
    aggregated: list[Code],
    vectors: np.ndarray,
    codebooks: list[Codebook],
    neighbor_threshold: float = 0.55,
    groups: dict[str, list[str]] | None = None,
    novelty_mode: str = "share",
) -> list[CoderMetrics]:
    """Score each codebook, then each named group, against ``aggregated``.

    Codebook weight is ``1 / ln(max(n_codes, median_n, 2))``. The floor of 2
    keeps the log defined when every codebook has a single code. A coder who
    owns a concept gets observation 1. Otherwise the observation is the log
    share of neighboring concepts they own, which lets a nearby code count
    without treating it as the same concept.

    Divergence is the Jensen-Shannon distance, ``sqrt(JSD)`` with the natural
    log. Its maximum is ``sqrt(ln 2)`` which is about 0.833. The paper prints
    this value times 100.

    ``novelty_mode`` chooses how Novelty's numerator is taken, and the two have
    different properties:

    ``"share"`` (default)
        Only concepts the row owns count, which is the paper's printed
        Algorithm 5 -- its numerator runs over ``c in csp_x``, and a coder's
        observation on a code they own is 1. Novelty is then each row's share
        of the uniquely-held mass, so the shares sum to 1 across coders. That
        makes it a clean decomposition, but it also means the *average* Novelty
        is fixed at ``1/n`` whatever the merging stage does, so Novelty cannot
        show a stage effect under this reading.

    ``"credited"``
        Observations are summed over every novel concept, so a row earns
        partial credit for a novel concept it neighbours but does not own. The
        totals then exceed 1 and vary with the merge, which is what the paper's
        published figures do: its eight coders sum to 128.88% at Condition 1
        and 118.29% at Condition 4, and its Table 5 reports a significant stage
        effect. Those numbers are unreachable under ``"share"``, so the paper's
        own results indicate this is the variant it ran -- in tension with the
        ``c in csp_x`` restriction it prints.
    """
    if novelty_mode not in ("share", "credited"):
        raise ValueError("novelty_mode must be 'share' or 'credited'")
    if not aggregated:
        return []
    coder_names = [codebook.name for codebook in codebooks]
    sizes = {codebook.name: len(codebook) for codebook in codebooks}
    weights = codebook_weights(sizes)
    distance = cosine_distance_matrix(vectors)
    neighbors = neighbor_indexes(distance, neighbor_threshold)
    observations = observation_matrix(aggregated, coder_names, neighbors)
    weight_vector = np.array([weights[name] for name in coder_names], dtype=float)
    score = observations.T @ weight_vector

    results = [
        _metrics_for(
            name=name,
            index=index,
            observations=observations,
            score=score,
            weight=float(weight_vector[index]),
            n_codes=sizes[name],
            consolidated=_consolidated_count(aggregated, {name}),
            novel_mask=_novel_mask(aggregated, coder_names, owners=None),
            owned_mask=_owned_mask(aggregated, {name}),
            baseline_weights=weight_vector,
            member_indexes=np.array([index]),
            novelty_mode=novelty_mode,
        )
        for index, name in enumerate(coder_names)
    ]

    groups = groups or {}
    name_to_index = {name: index for index, name in enumerate(coder_names)}
    for group_name, members in groups.items():
        missing = [member for member in members if member not in name_to_index]
        if missing:
            raise KeyError(f"group {group_name} names unknown coders: {', '.join(missing)}")
        member_indexes = np.array([name_to_index[member] for member in members], dtype=int)
        group_obs = _group_observations(aggregated, set(members), neighbors)
        stacked = np.vstack([observations, group_obs])
        member_set = set(members)
        results.append(
            _metrics_for(
                name=f"group: {group_name}",
                index=stacked.shape[0] - 1,
                observations=stacked,
                score=score,
                weight=0.0,
                n_codes=sum(sizes[member] for member in members),
                consolidated=_consolidated_count(aggregated, member_set),
                # The novel set is a property of the aggregated space, not of the
                # row being scored, so a group uses the same mask as a coder:
                # concepts exactly one coder found. Scoping it to the group
                # instead would make the numerator and denominator the same set
                # and every group would score exactly 1.
                novel_mask=_novel_mask(aggregated, coder_names, owners=None),
                owned_mask=_owned_mask(aggregated, member_set),
                baseline_weights=weight_vector,
                member_indexes=member_indexes,
                novelty_mode=novelty_mode,
            )
        )
    return results


def codebook_weights(sizes: dict[str, int]) -> dict[str, float]:
    """Down-weight large codebooks so flooding does not dominate the shared space."""
    positive = [size for size in sizes.values() if size > 0]
    if not positive:
        return {name: 0.0 for name in sizes}
    median = float(np.median(positive))
    weights: dict[str, float] = {}
    for name, size in sizes.items():
        if size <= 0:
            weights[name] = 0.0
            continue
        weights[name] = 1.0 / math.log(max(float(size), median, 2.0))
    return weights


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """JSD in nats. Zeros in either distribution are allowed."""
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    sum_p = float(p.sum())
    sum_q = float(q.sum())
    if sum_p <= 0.0 or sum_q <= 0.0:
        return 0.0
    p = p / sum_p
    q = q / sum_q
    mixture = 0.5 * (p + q)
    return 0.5 * _kl(p, mixture) + 0.5 * _kl(q, mixture)


def jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    return math.sqrt(jensen_shannon_divergence(left, right))


def observation_matrix(codes: list[Code], coders: list[str], neighbors: list[list[int]]) -> np.ndarray:
    """Rows are coders, columns are aggregated codes, values lie in ``[0, 1]``."""
    observations = np.zeros((len(coders), len(codes)), dtype=float)
    for code_index, code in enumerate(codes):
        neighbor_ids = neighbors[code_index]
        owned_neighbors: dict[str, int] = {}
        for neighbor in neighbor_ids:
            for owner in codes[neighbor].owners:
                owned_neighbors[owner] = owned_neighbors.get(owner, 0) + 1
        neighbor_count = len(neighbor_ids)
        for coder_index, coder in enumerate(coders):
            if coder in code.owners:
                observations[coder_index, code_index] = 1.0
            elif neighbor_count == 0:
                observations[coder_index, code_index] = 0.0
            else:
                count = owned_neighbors.get(coder, 0)
                observations[coder_index, code_index] = min(
                    1.0,
                    math.log(count + 1.0) / math.log(neighbor_count + 1.0),
                )
    return observations


def _group_observations(codes: list[Code], members: set[str], neighbors: list[list[int]]) -> np.ndarray:
    row = np.zeros(len(codes), dtype=float)
    for code_index, code in enumerate(codes):
        if code.owners & members:
            row[code_index] = 1.0
            continue
        neighbor_ids = neighbors[code_index]
        if not neighbor_ids:
            continue
        count = sum(1 for neighbor in neighbor_ids if codes[neighbor].owners & members)
        row[code_index] = min(1.0, math.log(count + 1.0) / math.log(len(neighbor_ids) + 1.0))
    return row


def _novel_mask(codes: list[Code], coders: list[str], owners: set[str] | None) -> np.ndarray:
    """A code is novel when exactly one coder identified it.

    ``owners`` narrows the mask to concepts held only inside that set. It is
    kept for callers that want a group-relative novel set, but note that
    scoring a group against its own novel set is vacuous: the numerator and
    denominator of Novelty coincide and the metric returns 1. Pass ``None`` to
    score any row against the shared novel set.
    """
    coder_set = set(coders)
    mask = np.zeros(len(codes), dtype=bool)
    for index, code in enumerate(codes):
        present = code.owners & coder_set
        if owners is None:
            mask[index] = len(present) == 1
        else:
            mask[index] = bool(present) and present <= owners
    return mask


def _owned_mask(codes: list[Code], owners: set[str]) -> np.ndarray:
    return np.array([bool(code.owners & owners) for code in codes], dtype=bool)


def _consolidated_count(codes: list[Code], owners: set[str]) -> int:
    return sum(1 for code in codes if code.owners & owners)


def _metrics_for(
    name: str,
    index: int,
    observations: np.ndarray,
    score: np.ndarray,
    weight: float,
    n_codes: int,
    consolidated: int,
    novel_mask: np.ndarray,
    owned_mask: np.ndarray,
    baseline_weights: np.ndarray,
    member_indexes: np.ndarray,
    novelty_mode: str = "share",
) -> CoderMetrics:
    observed = observations[index]
    total = float(score.sum())
    coverage = float(observed @ score) / total if total > 0 else 0.0

    removed = np.zeros_like(score)
    for member in member_indexes:
        removed += observations[int(member)] * float(baseline_weights[int(member)])
    baseline = score - removed
    baseline_total = float(baseline.sum())
    overlap = float(observed @ baseline) / baseline_total if baseline_total > 0 else 0.0

    # Novelty counts concepts nobody else identified. Under "share" only the
    # ones this row owns count, so the rows decompose the novel mass; under
    # "credited" a neighbouring row earns partial credit too.
    novel_mass = float(score[novel_mask].sum()) if novel_mask.any() else 0.0
    if novel_mass <= 0:
        novelty = 0.0
    elif novelty_mode == "credited":
        novelty = float((observed[novel_mask] * score[novel_mask]).sum()) / novel_mass
    else:
        contributed = novel_mask & owned_mask
        novelty = float(score[contributed].sum()) / novel_mass

    divergence = jensen_shannon_distance(baseline, observed)
    return CoderMetrics(
        coder=name,
        n_codes=n_codes,
        weight=weight,
        coverage=coverage,
        overlap=overlap,
        novelty=novelty,
        divergence=divergence,
        consolidated=consolidated,
    )


def _kl(source: np.ndarray, target: np.ndarray) -> float:
    mask = source > 0
    return float(np.sum(source[mask] * np.log(source[mask] / target[mask])))
