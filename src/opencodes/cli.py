"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from opencodes.embed import HashingEmbedder, SentenceTransformerEmbedder
from opencodes.io import dump_evaluation, load_codebooks
from opencodes.llm import OpenAIWriter, TemplateWriter
from opencodes.pipeline import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge inductive codebooks and score Coverage, Overlap, Novelty, and Divergence.",
    )
    parser.add_argument("codebooks", help="JSON file of named codebooks")
    parser.add_argument("--stage", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--lower", type=float, default=0.32, help="strict cosine-distance merge threshold")
    parser.add_argument("--upper", type=float, default=0.55, help="relaxed cosine-distance merge threshold")
    parser.add_argument(
        "--neighbor",
        type=float,
        default=None,
        help="cosine distance at which unmerged codes still count as neighbors (default: --upper)",
    )
    parser.add_argument(
        "--embedder",
        choices=("hashing", "sentence-transformers"),
        default="hashing",
    )
    parser.add_argument(
        "--model",
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="sentence-transformers model name",
    )
    parser.add_argument("--llm", choices=("template", "openai"), default="template")
    parser.add_argument("--output", default="", help="write metrics JSON to this path")
    args = parser.parse_args(argv)

    codebooks, groups, question = load_codebooks(args.codebooks)
    embedder = (
        HashingEmbedder()
        if args.embedder == "hashing"
        else SentenceTransformerEmbedder(args.model)
    )
    writer = TemplateWriter() if args.llm == "template" else OpenAIWriter()
    result = evaluate(
        codebooks,
        embedder=embedder,
        writer=writer,
        stage=args.stage,
        lower=args.lower,
        upper=args.upper,
        neighbor_threshold=args.neighbor,
        research_question=question,
        groups=groups,
    )

    history = " → ".join(str(size) for size in result.aggregated.sizes_after_stage)
    print(f"stage {result.aggregated.stage}: {history} codes in the aggregated space")
    print()
    header = f"{'coder':<22} {'codes':>6} {'in ACS':>6} {'cover':>8} {'overlap':>8} {'novel':>8} {'diverge':>8}"
    print(header)
    print("-" * len(header))
    rows = [metric.as_dict() for metric in result.metrics]
    for row in rows:
        print(
            f"{row['coder']:<22} {row['n_codes']:>6} {row['consolidated']:>6} "
            f"{row['coverage']:>8.3f} {row['overlap']:>8.3f} "
            f"{row['novelty']:>8.3f} {row['divergence']:>8.3f}"
        )
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(
            dump_evaluation(rows, len(result.aggregated), result.aggregated.stage),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
