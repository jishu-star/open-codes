# open-codes

A Python implementation of the evaluation method in Chen et al., *A Computational Method for Measuring “Open Codes” in Qualitative Analysis* (Findings of ACL 2026). The paper is [here](https://aclanthology.org/2026.findings-acl.2073/).

Inductive coding has no answer key. Two researchers, or a researcher and a model, can describe the same excerpt with different words and both be doing the job. This package does not score codes against a gold codebook. It merges every codebook into one shared concept space, then asks how each coder sits in that space.

## What gets computed

Each coder produces a **code space**: a set of labels, each with the excerpts it was applied to and, optionally, a definition. The union of those spaces is not yet comparable, because “user feedback” and “feedback from user” are two strings for one idea. A four-stage merge builds an **aggregated code space**:

1. **Exact labels.** Case and extra spaces are ignored. The same label from two coders becomes one concept, and both coders remain owners.
2. **Close labels.** Labels are embedded and clustered with average linkage. Pairs within cosine distance 0.32 are merged. The shorter label is kept.
3. **Definitions.** A writer drafts a definition from the label and its excerpts. Clustering is repeated on the label plus that definition, still at distance 0.32. Each merged cluster gets one new label and one definition.
4. **Relaxed, penalized merge.** Clustering repeats until a round merges nothing, now with two thresholds (0.32 and 0.55). A pair closer than 0.32 always merges. A pair farther than 0.55 never merges. In between, two penalties can refuse the merge: the excerpts barely overlap, or the merged concept would pile up many more excerpts than a typical code. Codes that almost merged stay **neighbors**. They are related, but they are not the same concept.

The paper’s printed penalty adds distance when excerpts overlap. That would make codes grounded in the same data harder to merge. The clustering heuristic that accompanies the paper does the reverse, and that is what this package implements: distance grows with the fraction of excerpts that do **not** overlap.

Each coder is then scored against the aggregated space. A large codebook is down-weighted by `1 / ln(max(n, median n, 2))`, so producing more labels does not automatically mean covering more ideas. Owning a concept counts as 1. Not owning it, but owning some of its neighbors, counts as the log share of those neighbors.

| Metric | Question it answers | Rises when |
| --- | --- | --- |
| Coverage | How much of the shared space does this coder reach? | The coder touches more of the concepts the team found |
| Overlap | How much of that reach is shared with the others? | The coder’s concepts are also someone else’s |
| Novelty | What share of the concepts nobody else found belongs to this coder? | The coder is the sole owner of those concepts |
| Divergence | How different is this coder’s emphasis from the leave-one-out team? | Mass sits on different concepts than the rest of the team |

Divergence is the Jensen–Shannon distance, `sqrt(JSD)` with the natural logarithm. It tops out near `0.833`, not `1`. The paper’s “Divergence %” column is this number times 100.

Read the four numbers together. High coverage plus high novelty can be a broad coder or a flood of near-duplicate labels. Low coverage, low overlap, and high divergence can be a distinct perspective or a coder who was looking at the wrong data. The metrics cannot see a concept that every coder missed.

## Install

```bash
pip install -e ".[dev]"
```

Semantic merges need a real embedding model. The default command uses a local hashing embedder so the install stays small; identical labels still merge, paraphrases usually do not.

```bash
pip install -e ".[embeddings]"
opencodes examples/toy_codebooks.json --embedder sentence-transformers --model all-MiniLM-L6-v2 --stage 2
```

The paper uses `mxbai-embed-large`. Pass that model name to `--model` if you have it. Stages 3 and 4 call a writer. `--llm template` records a mechanical definition and keeps the shortest label. `--llm openai` calls an OpenAI-compatible chat endpoint (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL` and `OPENAI_MODEL`). The paper used Gemma 3 27B at temperature 0.5 for this step and argues that coder rankings were stable across several models.

## Run

```bash
opencodes examples/toy_codebooks.json --stage 1
python examples/edge_cases.py
pytest
```

`--stage 1` only unions identical labels. `--stage 4` is the full method. A codebook file looks like this:

```json
{
  "research_question": "How did the community emerge?",
  "groups": {"human": ["human-a", "human-b"]},
  "codebooks": {
    "human-a": [
      {"label": "welcome newcomers", "examples": ["A teacher answered a first-time poster."]}
    ]
  }
}
```

`examples/edge_cases.py` rebuilds the paper’s diagnostic check on a toy geometry: a flooding coder (many extra labels) and a hallucinating coder (labels from an unrelated conversation). Expect the flood to raise coverage and novelty, and the unrelated coder to lower coverage and overlap while raising divergence. Those are directions, not the paper’s published percentages.

## Layout

- `opencodes.aggregate` builds the aggregated code space.
- `opencodes.metrics` computes the four scores.
- `opencodes.pipeline.evaluate` runs both.
- Embedders and writers are arguments, so a test can pin every vector and never call a model.

The authors’ full coding system is [CHAIR](https://github.com/CIVITAS-John/CHAIR). This repository implements the measurement method from the paper, not that system’s coding interface.
