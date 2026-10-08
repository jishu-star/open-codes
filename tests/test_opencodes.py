"""Controlled checks for merging and for the four metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opencodes.aggregate import aggregate_codebooks
from opencodes.cluster import (
    apply_example_penalty,
    clusters_with_penalties,
    shared_example_fraction,
    unique_example_fraction,
)
from opencodes.embed import HashingEmbedder, TableEmbedder, cosine_distance_matrix
from opencodes.metrics import codebook_weights, evaluate_codebooks, jensen_shannon_distance
from opencodes.pipeline import evaluate
from opencodes.types import Code, Codebook


def _book(name: str, labels: list[str], examples: dict[str, list[str]] | None = None) -> Codebook:
    examples = examples or {}
    return Codebook(
        name=name,
        codes=[Code(label=label, examples=set(examples.get(label, []))) for label in labels],
    )


def _unit(x: float, y: float, z: float = 0.0) -> list[float]:
    vector = np.array([x, y, z], dtype=float)
    return (vector / np.linalg.norm(vector)).tolist()


def test_identical_strings_have_zero_cosine_distance():
    vectors = HashingEmbedder().embed(["peer support", "peer support", "pricing complaint"])
    distance = cosine_distance_matrix(vectors)
    # Normalising then taking a dot product leaves float noise of order 1e-16,
    # so identity is asserted to tolerance rather than exactly.
    assert distance[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert distance[0, 2] > 0.2


def test_stage1_unions_labels_ignoring_case():
    result = aggregate_codebooks(
        [
            _book("ann", ["Peer Support"]),
            _book("bo", ["peer   support"]),
        ],
        embedder=TableEmbedder({"peer support": [1.0, 0.0]}),
        stage=1,
    )
    assert len(result) == 1
    assert result.codes[0].owners == {"ann", "bo"}


def test_stage2_merges_close_labels_and_keeps_the_shorter_one():
    table = {
        "user feedback": _unit(1, 0),
        "feedback from user": _unit(0.95, math.sqrt(1 - 0.95**2)),
        "pricing": _unit(0, 1),
    }
    result = aggregate_codebooks(
        [_book("ann", ["feedback from user", "pricing"]), _book("bo", ["user feedback"])],
        embedder=TableEmbedder(table),
        stage=2,
        lower=0.32,
    )
    labels = {code.label for code in result.codes}
    assert labels == {"user feedback", "pricing"}
    feedback = next(code for code in result.codes if code.label == "user feedback")
    assert feedback.owners == {"ann", "bo"}
    assert "feedback from user" in feedback.alternatives


def test_example_penalty_blocks_a_gray_zone_merge_of_disjoint_excerpts():
    # Cosine distance 0.40 sits between the paper's 0.32 and 0.55 cutoffs.
    left = np.array(_unit(1, 0))
    right = np.array(_unit(0.6, 0.8))
    distance = cosine_distance_matrix(np.vstack([left, right]))
    assert 0.32 < distance[0, 1] < 0.55

    separated = clusters_with_penalties(distance, [{"msg-1"}, {"msg-2"}], lower=0.32, upper=0.55)
    assert sorted(len(group) for group in separated) == [1, 1]

    joined = clusters_with_penalties(distance, [{"msg-1"}, {"msg-1"}], lower=0.32, upper=0.55)
    assert len(joined) == 1 and len(joined[0]) == 2


def test_unique_example_fraction_is_zero_when_excerpts_match():
    assert unique_example_fraction({"a", "b"}, {"b", "a"}) == 0.0
    assert unique_example_fraction({"a"}, {"b"}) == 1.0


def test_large_codebooks_receive_a_smaller_weight():
    weights = codebook_weights({"small": 2, "large": 40})
    assert weights["small"] > weights["large"]
    assert weights["small"] == 1 / math.log(21)


def test_identical_codebooks_agree_and_add_nothing_unique():
    shared = TableEmbedder({"alpha": [1.0, 0.0], "beta": [0.0, 1.0]})
    books = [_book("ann", ["alpha", "beta"]), _book("bo", ["alpha", "beta"])]
    result = evaluate(books, embedder=shared, stage=1, neighbor_threshold=0.55)
    by_name = {metric.coder: metric for metric in result.metrics}
    for metric in by_name.values():
        assert metric.coverage == 1.0
        assert metric.overlap == 1.0
        assert metric.novelty == 0.0
        assert metric.divergence == 0.0


def test_a_coder_on_unrelated_codes_covers_less_and_diverges_more():
    table = TableEmbedder(
        {
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
            "delta": [0.0, 0.0, -1.0],
        }
    )
    books = [
        _book("ann", ["alpha", "beta"]),
        _book("bo", ["alpha", "beta"]),
        _book("hallucinator", ["gamma", "delta"]),
    ]
    result = evaluate(books, embedder=table, stage=1, neighbor_threshold=0.2)
    by_name = {metric.coder: metric for metric in result.metrics}
    assert by_name["hallucinator"].coverage < by_name["ann"].coverage
    assert by_name["hallucinator"].overlap < by_name["ann"].overlap
    assert by_name["hallucinator"].novelty > by_name["ann"].novelty
    assert by_name["hallucinator"].divergence > by_name["ann"].divergence


def test_flooding_raises_coverage_and_novelty_above_the_humans():
    table = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
    for index in range(6):
        angle = (index + 1) * 0.7
        table[f"extra-{index}"] = [math.cos(angle + 2), math.sin(angle + 2)]
    books = [
        _book("ann", ["alpha", "beta"]),
        _book("bo", ["alpha", "beta"]),
        _book("flood", ["alpha", "beta", *[f"extra-{index}" for index in range(6)]]),
    ]
    result = evaluate(books, embedder=TableEmbedder(table), stage=1, neighbor_threshold=0.05)
    by_name = {metric.coder: metric for metric in result.metrics}
    assert by_name["flood"].coverage > by_name["ann"].coverage
    assert by_name["flood"].novelty > by_name["ann"].novelty
    assert by_name["flood"].weight < by_name["ann"].weight


def test_one_neighbor_is_partial_reach_not_ownership():
    # Three neighbors: owning one of them is log(2)/log(4) = 0.5, not a full hit.
    codes = [
        Code(label="center", owners={"ann"}),
        Code(label="near-ann", owners={"ann"}),
        Code(label="near-bo", owners={"bo"}),
        Code(label="near-cy", owners={"cy"}),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.8, 0.0, 0.6],
            [0.8, -0.6, 0.0],
        ]
    )
    books = [
        _book("ann", ["center", "near-ann"]),
        _book("bo", ["near-bo"]),
        _book("cy", ["near-cy"]),
    ]
    metrics = evaluate_codebooks(codes, vectors, books, neighbor_threshold=0.5)
    # Sanity: the center code really does have three neighbors under this cutoff.
    distance = cosine_distance_matrix(vectors)
    assert sum(distance[0, 1:] <= 0.5) == 3
    by_name = {metric.coder: metric for metric in metrics}
    assert by_name["ann"].coverage > by_name["bo"].coverage


def test_group_covers_the_union_of_its_members():
    table = TableEmbedder({"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "gamma": [-1.0, 0.0]})
    books = [
        _book("ann", ["alpha"]),
        _book("bo", ["beta"]),
        _book("machine", ["gamma"]),
    ]
    result = evaluate(
        books,
        embedder=table,
        stage=1,
        neighbor_threshold=0.1,
        groups={"human": ["ann", "bo"]},
    )
    by_name = {metric.coder: metric for metric in result.metrics}
    assert by_name["group: human"].coverage > by_name["ann"].coverage
    assert by_name["group: human"].consolidated == 2


def test_jsd_is_zero_for_the_same_distribution_and_bounded():
    same = np.array([0.2, 0.8, 0.0])
    assert jensen_shannon_distance(same, same) == 0.0
    opposite = jensen_shannon_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert 0.8 < opposite <= math.sqrt(math.log(2)) + 1e-9


def test_stage4_merge_uses_shared_excerpts():
    table = {
        "user feedback": _unit(1, 0),
        "comments from users": _unit(0.6, 0.8),
    }
    shared = {"thread-4"}
    books = [
        Codebook("ann", [Code(label="user feedback", examples=set(shared))]),
        Codebook("bo", [Code(label="comments from users", examples=set(shared))]),
    ]
    merged = aggregate_codebooks(books, embedder=TableEmbedder(table), stage=4, lower=0.32, upper=0.55)
    disjoint = aggregate_codebooks(
        [
            Codebook("ann", [Code(label="user feedback", examples={"thread-4"})]),
            Codebook("bo", [Code(label="comments from users", examples={"thread-9"})]),
        ],
        embedder=TableEmbedder(table),
        stage=4,
        lower=0.32,
        upper=0.55,
    )
    assert len(merged) == 1
    assert len(disjoint) == 2


def test_group_novelty_is_a_share_not_a_constant():
    """A group scored against its own novel set would always return 1.

    Regression test: Novelty's novel set is a property of the aggregated space,
    so a group row must be scored against the concepts exactly one coder found,
    the same set an individual coder is scored against. Scoping the set to the
    group makes the numerator and denominator coincide and every group reports
    100%, which is what this package used to do.
    """
    table = TableEmbedder(
        {
            "alpha": _unit(1, 0),
            "beta": _unit(0, 1),
            "gamma": _unit(-1, 0),
            "delta": _unit(0, -1),
        }
    )
    # Every code is held by exactly one coder, so all four are novel. The human
    # group owns two of them and the machine group the other two.
    books = [
        _book("ann", ["alpha"]),
        _book("bo", ["beta"]),
        _book("m1", ["gamma"]),
        _book("m2", ["delta"]),
    ]
    result = evaluate(
        books,
        embedder=table,
        stage=1,
        neighbor_threshold=0.1,
        groups={"human": ["ann", "bo"], "machine": ["m1", "m2"]},
    )
    by_name = {metric.coder: metric for metric in result.metrics}
    human = by_name["group: human"].novelty
    machine = by_name["group: machine"].novelty

    # Neither group holds all of the novel mass, so neither can score 1.
    assert 0.0 < human < 1.0
    assert 0.0 < machine < 1.0
    # The groups partition the coders and every novel code has one owner, so
    # their shares exhaust the novel mass.
    assert human + machine == pytest.approx(1.0)


def test_individual_novelty_shares_sum_to_one():
    """Novelty is a share of the uniquely-held mass, so the coders exhaust it."""
    table = TableEmbedder(
        {"alpha": _unit(1, 0), "beta": _unit(0, 1), "gamma": _unit(-1, 0)}
    )
    books = [_book("ann", ["alpha"]), _book("bo", ["beta"]), _book("cy", ["gamma"])]
    result = evaluate(books, embedder=table, stage=1, neighbor_threshold=0.1)
    total = sum(metric.novelty for metric in result.metrics)
    assert total == pytest.approx(1.0)


def test_penalty_mode_paper_reverses_the_example_penalty():
    """The printed Algorithm 1 and the shipped heuristic move merging opposite ways.

    Two codes sharing every excerpt get no penalty under the heuristic (nothing
    is unique) but the maximum penalty under the paper's printed formula (the
    overlap is total).
    """
    distance = np.array([[0.0, 0.45], [0.45, 0.0]])
    examples = [{"e1", "e2"}, {"e1", "e2"}]

    heuristic = apply_example_penalty(distance, examples, 0.32, 0.55)
    paper = apply_example_penalty(
        distance, examples, 0.32, 0.55, penalty_mode="paper"
    )
    assert heuristic[0, 1] == pytest.approx(0.45)
    assert paper[0, 1] == pytest.approx(0.45 + (0.55 - 0.32))
    assert paper[0, 1] > heuristic[0, 1]

    with pytest.raises(ValueError):
        apply_example_penalty(distance, examples, 0.32, 0.55, penalty_mode="nope")


def test_shared_and_unique_example_fractions_are_complementary():
    left, right = {"a", "b", "c"}, {"b", "c", "d"}
    assert shared_example_fraction(left, right) + unique_example_fraction(
        left, right
    ) == pytest.approx(1.0)
    assert shared_example_fraction(set(), set()) == 0.0


def test_novelty_mode_credited_exceeds_the_share_and_is_not_a_decomposition():
    """Credited Novelty lets a neighbouring row earn partial credit.

    The share reading sums to 1 across coders, which pins its mean at 1/n and
    makes it unable to register a change in the merge. The credited reading
    sums above 1, so its mean can move. The paper's published Novelty figures
    sum to 128.88%, so only the credited reading can reproduce them.
    """
    # Three codes, one per coder, all mutual neighbours under a loose cutoff.
    codes = [
        Code(label="alpha", owners={"ann"}),
        Code(label="beta", owners={"bo"}),
        Code(label="gamma", owners={"cy"}),
    ]
    vectors = np.array([_unit(1, 0, 0), _unit(0.9, 0.44, 0), _unit(0.9, 0, 0.44)])
    books = [_book("ann", ["alpha"]), _book("bo", ["beta"]), _book("cy", ["gamma"])]

    share = evaluate_codebooks(codes, vectors, books, neighbor_threshold=0.9)
    credited = evaluate_codebooks(
        codes, vectors, books, neighbor_threshold=0.9, novelty_mode="credited"
    )
    share_total = sum(metric.novelty for metric in share)
    credited_total = sum(metric.novelty for metric in credited)

    assert share_total == pytest.approx(1.0)
    assert credited_total > share_total
    # Every row gains, because each owns one novel code and neighbours the rest.
    by_share = {m.coder: m.novelty for m in share}
    for metric in credited:
        assert metric.novelty > by_share[metric.coder]

    with pytest.raises(ValueError):
        evaluate_codebooks(codes, vectors, books, novelty_mode="nope")
