"""Measure inductive open codes without assuming a ground-truth codebook."""

from opencodes.aggregate import AggregateResult, aggregate_codebooks
from opencodes.metrics import CoderMetrics, evaluate_codebooks
from opencodes.pipeline import Evaluation, evaluate
from opencodes.types import Code, Codebook

__all__ = [
    "AggregateResult",
    "Code",
    "Codebook",
    "CoderMetrics",
    "Evaluation",
    "aggregate_codebooks",
    "evaluate",
    "evaluate_codebooks",
]

__version__ = "0.1.0"
