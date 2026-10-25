"""Self-check for the scoring math (no databases needed): python -m eval.test_metrics"""
from app.retrieval.retriever import rrf_merge
from eval.ablation import relevant_ids, score

# RRF: an id ranked well by two legs beats one ranked first by a single leg
assert rrf_merge([["a", "b"], ["c", "b"]])[0] == "b"
# a down-weighted list loses the tie it would otherwise win
assert rrf_merge([["a"], ["g"]], weights=[1.0, 0.5])[0] == "a"
assert rrf_merge([["a"], ["g"]], weights=[0.5, 1.0])[0] == "g"

assert score(["x", "r1", "y"], {"r1", "r2"}, k=2) == (1.0, 0.5, 0.5)
assert score(["x", "y", "r1"], {"r1"}, k=2) == (0.0, 0.0, 1 / 3)  # RR counts past k
assert score([], {"r1"}, k=8) == (0.0, 0.0, 0.0)

chunks = {"c1": "the log-structured  merge tree", "c2": "b-trees"}
assert relevant_ids(chunks, ["Log-Structured\nMerge"]) == {"c1"}  # case + whitespace

print("ok")
