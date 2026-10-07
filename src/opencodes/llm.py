"""Writers that turn labels and excerpts into a definition, and merged codes into one concept.

The published method asks a language model to do both jobs. The template writer
keeps the pipeline runnable with no API key. It does not imitate model judgment;
it only records the label and a short trace of the excerpts.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

from opencodes.types import shortest_label


class DefinitionWriter(Protocol):
    def write_definition(self, label: str, examples: list[str], research_question: str) -> str:
        """Return a definition for one code."""

    def write_merge(
        self,
        labels: list[str],
        definitions: list[str],
        examples: list[str],
        research_question: str,
    ) -> tuple[str, str]:
        """Return ``(label, definition)`` for a cluster of codes that will be merged."""


class TemplateWriter:
    """Offline stand-in. Merges keep the shortest label and join existing definitions."""

    def write_definition(self, label: str, examples: list[str], research_question: str) -> str:
        del research_question
        quoted = _quote(examples)
        if quoted:
            return f"Criteria for '{label}', grounded in: {quoted}"
        return f"Criteria for '{label}'."

    def write_merge(
        self,
        labels: list[str],
        definitions: list[str],
        examples: list[str],
        research_question: str,
    ) -> tuple[str, str]:
        del research_question
        label = shortest_label(labels)
        parts = [item for item in definitions if item]
        if not parts:
            parts = [self.write_definition(label, examples, "")]
        definition = " ".join(dict.fromkeys(parts))
        return label, definition


class OpenAIWriter:
    """Chat-completions client for an OpenAI-compatible endpoint.

    Reads ``OPENAI_API_KEY``. Optional ``OPENAI_BASE_URL`` (default
    ``https://api.openai.com/v1``) and ``OPENAI_MODEL`` (default ``gpt-4.1-mini``).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.5,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def write_definition(self, label: str, examples: list[str], research_question: str) -> str:
        system = (
            "You are an expert in thematic analysis clarifying the criteria of qualitative codes. "
            "Write one clear, generalizable sentence of criteria. Do not introduce details that are "
            "not supported by the quotes. Reply with only the criteria sentence."
        )
        user = _definition_user(label, examples, research_question)
        return _first_line(self._complete(system, user))

    def write_merge(
        self,
        labels: list[str],
        definitions: list[str],
        examples: list[str],
        research_question: str,
    ) -> tuple[str, str]:
        system = (
            "You are an expert in thematic analysis. Several codes describe one concept. "
            "Write a single criteria sentence covering the input concepts, then a short label. "
            "Reply in exactly two lines:\n"
            "Criteria: ...\n"
            "Label: ..."
        )
        user = _merge_user(labels, definitions, examples, research_question)
        text = self._complete(system, user)
        return _parse_merge(text, labels)

    def _complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"]).strip()


def _quote(examples: list[str], limit: int = 5) -> str:
    unique: list[str] = []
    for example in examples:
        text = example.split("|||", 1)[-1].strip()
        if text and text not in unique:
            unique.append(text)
    unique.sort(key=len, reverse=True)
    return "; ".join(unique[:limit])


def _definition_user(label: str, examples: list[str], research_question: str) -> str:
    quotes = "\n".join(f"- {example.split('|||', 1)[-1].strip()}" for example in examples[:8])
    question = f"\nResearch question: {research_question}" if research_question else ""
    return f"Label: {label}{question}\nQuotes:\n{quotes or '- (none)'}"


def _merge_user(
    labels: list[str],
    definitions: list[str],
    examples: list[str],
    research_question: str,
) -> str:
    lines = [f"Concepts: {', '.join(labels)}"]
    if research_question:
        lines.append(f"Research question: {research_question}")
    for definition in definitions:
        lines.append(f"- {definition}")
    quoted = _quote(examples)
    if quoted:
        lines.append(f"Quotes: {quoted}")
    return "\n".join(lines)


def _first_line(text: str) -> str:
    line = text.strip().splitlines()[0].strip()
    if line.lower().startswith("criteria:"):
        line = line.split(":", 1)[1].strip()
    return line


def _parse_merge(text: str, labels: list[str]) -> tuple[str, str]:
    criteria = ""
    label = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("criteria:"):
            criteria = line.split(":", 1)[1].strip()
        elif line.lower().startswith("label:") or line.lower().startswith("phrase:"):
            label = line.split(":", 1)[1].strip().rstrip(".")
    if not label:
        label = shortest_label(labels)
    if not criteria:
        criteria = text.strip() or f"Criteria for '{label}'."
    return label, criteria
