"""JSON codebook files."""

from __future__ import annotations

import json
from pathlib import Path

from opencodes.types import Code, Codebook


def load_codebooks(path: str | Path) -> tuple[list[Codebook], dict[str, list[str]], str]:
    """Read a codebook file.

    The file is a JSON object::

        {
          "research_question": "...",
          "groups": {"human": ["human-a", "human-b"]},
          "codebooks": {
            "human-a": [
              {"label": "peer support", "examples": ["..."], "definitions": ["..."]}
            ]
          }
        }
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "codebooks" not in payload:
        raise ValueError("codebook file needs a 'codebooks' object")
    codebooks = [
        Codebook(name=name, codes=[_code(item, name) for item in items])
        for name, items in payload["codebooks"].items()
    ]
    groups = {name: list(members) for name, members in payload.get("groups", {}).items()}
    question = str(payload.get("research_question", ""))
    return codebooks, groups, question


def dump_evaluation(metrics: list[dict], aggregated_size: int, stage: int) -> str:
    return json.dumps(
        {"stage": stage, "aggregated_codes": aggregated_size, "coders": metrics},
        indent=2,
    )


def _code(item: dict, owner: str) -> Code:
    if "label" not in item:
        raise ValueError(f"code in {owner} is missing a label")
    definitions = item.get("definitions") or []
    if isinstance(definitions, str):
        definitions = [definitions]
    single = item.get("definition")
    if single and single not in definitions:
        definitions = [single, *definitions]
    examples = item.get("examples") or []
    return Code(
        label=str(item["label"]).strip(),
        definitions=[str(definition) for definition in definitions],
        examples={str(example) for example in examples},
        owners={owner},
    )
