"""Codebooks produced by one inductive coder."""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize_label(label: str) -> str:
    return " ".join(label.casefold().split())


@dataclass
class Code:
    """One inductive code.

    ``examples`` are the data excerpts the code was applied to. ``owners`` are
    the coders who contributed a code that landed in this concept after merging.
    ``alternatives`` keeps the other labels that were folded in.
    """

    label: str
    definitions: list[str] = field(default_factory=list)
    examples: set[str] = field(default_factory=set)
    owners: set[str] = field(default_factory=set)
    alternatives: list[str] = field(default_factory=list)

    def copy(self) -> Code:
        return Code(
            label=self.label,
            definitions=list(self.definitions),
            examples=set(self.examples),
            owners=set(self.owners),
            alternatives=list(self.alternatives),
        )


@dataclass
class Codebook:
    """The code space of a single coder (or a named group of coders)."""

    name: str
    codes: list[Code] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.codes)


def code_text(code: Code, use_definition: bool) -> str:
    """Text embedded for a code. Stage 2 uses the label; later stages add a definition."""
    if use_definition and code.definitions:
        return f"Label: {code.label}\nDefinition: {code.definitions[0]}"
    return code.label


def shortest_label(labels: list[str]) -> str:
    return min(labels, key=lambda label: (len(label), label.casefold()))


def merge_codes(codes: list[Code], label: str | None = None, definition: str | None = None) -> Code:
    """Fold several codes into one concept.

    Stage 2 adopts the shorter label. Later stages may pass a label and definition
    written by the language model.
    """
    if not codes:
        raise ValueError("cannot merge an empty group of codes")
    chosen = label if label else shortest_label([code.label for code in codes])
    definitions: list[str] = []
    if definition:
        definitions = [definition]
    else:
        for code in codes:
            for item in code.definitions:
                if item not in definitions:
                    definitions.append(item)
    examples: set[str] = set()
    owners: set[str] = set()
    alternatives: list[str] = []
    for code in codes:
        examples |= set(code.examples)
        owners |= set(code.owners)
        for candidate in [code.label, *code.alternatives]:
            if normalize_label(candidate) == normalize_label(chosen):
                continue
            if candidate not in alternatives:
                alternatives.append(candidate)
    return Code(
        label=chosen,
        definitions=definitions,
        examples=examples,
        owners=owners,
        alternatives=alternatives,
    )
