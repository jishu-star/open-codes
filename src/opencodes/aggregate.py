"""Four-stage construction of an aggregated code space."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opencodes.cluster import clusters_at_threshold, clusters_with_penalties
from opencodes.embed import Embedder, cosine_distance_matrix
from opencodes.llm import DefinitionWriter, TemplateWriter
from opencodes.types import Code, Codebook, code_text, merge_codes, normalize_label


@dataclass
class AggregateResult:
    """Merged code space and the vectors used to place neighbors."""

    codes: list[Code]
    vectors: np.ndarray
    stage: int
    sizes_after_stage: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.codes)


def aggregate_codebooks(
    codebooks: list[Codebook],
    embedder: Embedder,
    writer: DefinitionWriter | None = None,
    stage: int = 4,
    lower: float = 0.32,
    upper: float = 0.55,
    research_question: str = "",
    max_rounds: int = 12,
) -> AggregateResult:
    """Merge codebooks through the stages requested by ``stage`` (1 through 4).

    1. Union of labels, matched after case-folding.
    2. Hierarchical merge of very similar labels. The shorter label is kept.
       Strict cosine distance is ``lower`` (0.32 in the paper).
    3. A definition is written from each code's label and excerpts, then codes
       are merged again on the label-plus-definition embedding. A writer names
       each merged concept.
    4. The same label-plus-definition merge is repeated with the two-threshold
       penalty cut (``lower``, ``upper``) until a round merges nothing.
    """
    if stage not in (1, 2, 3, 4):
        raise ValueError("stage must be 1, 2, 3, or 4")
    if lower < 0 or upper < lower:
        raise ValueError("thresholds must satisfy 0 <= lower <= upper")

    writer = writer or TemplateWriter()
    codes = _union_by_label(_stamp_owners(codebooks))
    history = [len(codes)]
    vectors = _embed(codes, embedder, use_definition=False)

    if stage >= 2:
        codes = _merge_groups(
            codes,
            clusters_at_threshold(cosine_distance_matrix(vectors), lower),
            writer=None,
            research_question="",
        )
        history.append(len(codes))
        vectors = _embed(codes, embedder, use_definition=False)

    if stage >= 3:
        codes = _ensure_definitions(codes, writer, research_question)
        vectors = _embed(codes, embedder, use_definition=True)
        groups = clusters_at_threshold(cosine_distance_matrix(vectors), lower)
        codes = _merge_groups(codes, groups, writer=writer, research_question=research_question)
        history.append(len(codes))

    if stage >= 4:
        for _ in range(max_rounds):
            before = len(codes)
            codes = _ensure_definitions(codes, writer, research_question)
            vectors = _embed(codes, embedder, use_definition=True)
            distance = cosine_distance_matrix(vectors)
            groups = clusters_with_penalties(
                distance,
                [set(code.examples) for code in codes],
                lower,
                upper,
            )
            codes = _merge_groups(codes, groups, writer=writer, research_question=research_question)
            history.append(len(codes))
            if len(codes) == before:
                break

    vectors = _embed(codes, embedder, use_definition=stage >= 3 and _any_definition(codes))
    return AggregateResult(codes=codes, vectors=vectors, stage=stage, sizes_after_stage=history)


def _stamp_owners(codebooks: list[Codebook]) -> list[Code]:
    stamped: list[Code] = []
    for codebook in codebooks:
        for code in codebook.codes:
            copied = code.copy()
            copied.owners.add(codebook.name)
            stamped.append(copied)
    return stamped


def _union_by_label(codes: list[Code]) -> list[Code]:
    grouped: dict[str, list[Code]] = {}
    order: list[str] = []
    for code in codes:
        key = normalize_label(code.label)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(code)
    return [merge_codes(grouped[key]) for key in order]


def _ensure_definitions(codes: list[Code], writer: DefinitionWriter, research_question: str) -> list[Code]:
    written: list[Code] = []
    for code in codes:
        if code.definitions:
            written.append(code)
            continue
        copied = code.copy()
        copied.definitions = [
            writer.write_definition(copied.label, _example_list(copied), research_question)
        ]
        written.append(copied)
    return written


def _merge_groups(
    codes: list[Code],
    groups: list[list[int]],
    writer: DefinitionWriter | None,
    research_question: str,
) -> list[Code]:
    merged: list[Code] = []
    used_labels: set[str] = set()
    for group in groups:
        members = [codes[index] for index in group]
        if len(members) == 1 or writer is None:
            code = merge_codes(members)
        else:
            label, definition = writer.write_merge(
                [member.label for member in members],
                [definition for member in members for definition in member.definitions],
                [example for member in members for example in _example_list(member)],
                research_question,
            )
            code = merge_codes(members, label=label, definition=definition)
        code.label = _unique_label(code.label, used_labels)
        used_labels.add(normalize_label(code.label))
        merged.append(code)
    return merged


def _unique_label(label: str, used: set[str]) -> str:
    if normalize_label(label) not in used:
        return label
    suffix = 2
    while normalize_label(f"{label} ({suffix})") in used:
        suffix += 1
    return f"{label} ({suffix})"


def _embed(codes: list[Code], embedder: Embedder, use_definition: bool) -> np.ndarray:
    if not codes:
        return np.zeros((0, 1), dtype=float)
    return embedder.embed([code_text(code, use_definition) for code in codes])


def _example_list(code: Code) -> list[str]:
    return sorted(code.examples, key=lambda item: (-len(item), item))


def _any_definition(codes: list[Code]) -> bool:
    return any(code.definitions for code in codes)
