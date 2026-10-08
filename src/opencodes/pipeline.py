"""Load codebooks and run aggregation plus the four metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opencodes.aggregate import AggregateResult, aggregate_codebooks
from opencodes.embed import Embedder
from opencodes.llm import DefinitionWriter
from opencodes.metrics import CoderMetrics, evaluate_codebooks
from opencodes.types import Codebook


@dataclass
class Evaluation:
    aggregated: AggregateResult
    metrics: list[CoderMetrics]
    research_question: str = ""
    groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def vectors(self) -> np.ndarray:
        return self.aggregated.vectors


def evaluate(
    codebooks: list[Codebook],
    embedder: Embedder,
    writer: DefinitionWriter | None = None,
    stage: int = 4,
    lower: float = 0.32,
    upper: float = 0.55,
    neighbor_threshold: float | None = None,
    research_question: str = "",
    groups: dict[str, list[str]] | None = None,
    penalty_mode: str = "heuristic",
    novelty_mode: str = "share",
) -> Evaluation:
    """Build the aggregated code space and score every codebook against it.

    ``penalty_mode`` selects stage 4's example penalty and ``novelty_mode`` how
    Novelty's numerator is taken; see
    :func:`opencodes.cluster.apply_example_penalty` and
    :func:`opencodes.metrics.evaluate_codebooks`.
    """
    aggregated = aggregate_codebooks(
        codebooks,
        embedder=embedder,
        writer=writer,
        stage=stage,
        lower=lower,
        upper=upper,
        research_question=research_question,
        penalty_mode=penalty_mode,
    )
    metrics = evaluate_codebooks(
        aggregated.codes,
        aggregated.vectors,
        codebooks,
        neighbor_threshold=upper if neighbor_threshold is None else neighbor_threshold,
        groups=groups,
        novelty_mode=novelty_mode,
    )
    return Evaluation(
        aggregated=aggregated,
        metrics=metrics,
        research_question=research_question,
        groups=dict(groups or {}),
    )
