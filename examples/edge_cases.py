"""Show how the four metrics react to flooding and to codes from the wrong data.

This uses a fixed geometry instead of a language model, so the direction of
each effect is visible without API keys. It is a miniature of the paper's
second experiment, not a reproduction of its numbers.
"""

from __future__ import annotations

import math

from opencodes.embed import TableEmbedder
from opencodes.pipeline import evaluate
from opencodes.types import Code, Codebook


def _axis(index: int) -> list[float]:
    vector = [0.0, 0.0, 0.0, 0.0]
    vector[index] = 1.0
    return vector


def _book(name: str, labels: list[str]) -> Codebook:
    return Codebook(name=name, codes=[Code(label=label, examples={f"{name}:{label}"}) for label in labels])


def _table() -> TableEmbedder:
    shared = {
        "welcome newcomers": _axis(0),
        "teachers share lesson files": _axis(1),
    }
    flood = {
        f"flood-{index}": [math.cos(index), math.sin(index), 0.2 * index, 0.0]
        for index in range(8)
    }
    hallucinated = {
        "recipe substitutions": _axis(2),
        "restaurant reservations": _axis(3),
    }
    return TableEmbedder({**shared, **flood, **hallucinated})


def _print(title: str, books: list[Codebook]) -> None:
    result = evaluate(books, embedder=_table(), stage=1, neighbor_threshold=0.15)
    print(title)
    print(f"  aggregated concepts: {len(result.aggregated)}")
    for metric in result.metrics:
        print(
            f"  {metric.coder:<16} cover={metric.coverage:.3f}  "
            f"overlap={metric.overlap:.3f}  novel={metric.novelty:.3f}  "
            f"diverge={metric.divergence:.3f}  codes={metric.n_codes}"
        )
    print()


def main() -> None:
    humans = [
        _book("human-a", ["welcome newcomers", "teachers share lesson files"]),
        _book("human-b", ["welcome newcomers", "teachers share lesson files"]),
    ]
    baseline = humans + [_book("machine", ["welcome newcomers", "teachers share lesson files"])]
    flooding = humans + [
        _book(
            "machine",
            ["welcome newcomers", "teachers share lesson files", *[f"flood-{index}" for index in range(8)]],
        )
    ]
    hallucinating = humans + [_book("machine", ["recipe substitutions", "restaurant reservations"])]
    _print("baseline", baseline)
    _print("flooding", flooding)
    _print("hallucinating", hallucinating)


if __name__ == "__main__":
    main()
