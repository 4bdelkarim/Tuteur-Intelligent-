#!/usr/bin/env python3
"""
calibrate_refusal_gate.py — calibre le seuil de refus sur un golden dataset,
en ne faisant QUE du retrieval (pas de generation) -- rapide, pour iterer sur
le seuil sans attendre un run complet.

NOTE (2026-08-06) : should_refuse() / REFUSAL_THRESHOLD ont ete supprimes car
le score de retrieval (RRF/cosine) ne discrimine pas answerable/unanswerable.
Ce script reste utile pour calibrer should_refuse_reranker() — le seuil optimal
est a reporter dans RERANKER_REFUSAL_THRESHOLD (refusal_gate.py).

Usage :
  python calibrate_refusal_gate.py eval/golden_dataset.json
"""

import sys

from .evaluate import load_dataset
from ..core.query_processing import process_query
from ..core.retriever import retrieve, merge_dedup
from ..core.refusal_gate import calibrate


def collect_scores(dataset_path, k=4):
    items = load_dataset(dataset_path)
    scored = []
    for item in items:
        if item["category"] is None:
            continue   # pas de categorie -> impossible de savoir si le refus etait attendu
        proc = process_query(item["question"])
        queries = proc["sub_queries"] or [proc["rewritten"]]
        hits = []
        for q in queries:
            hits = merge_dedup(hits, retrieve(q, k=k))
        top_score = hits[0]["dist"] if hits else None
        if top_score is not None:
            scored.append((top_score, item["category"] == "unanswerable"))
    return scored


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/golden_dataset.json"
    scored = collect_scores(path)

    print(f"{len(scored)} questions avec un score exploitable")
    print("\nDistribution des scores (top hit) par categorie :")
    for is_unans in (False, True):
        vals = sorted(s for s, u in scored if u == is_unans)
        label = "unanswerable" if is_unans else "answerable"
        if vals:
            print(f"  {label:<12} n={len(vals):<3} min={vals[0]:.3f}  "
                  f"med={vals[len(vals)//2]:.3f}  max={vals[-1]:.3f}")
        else:
            print(f"  {label:<12} (aucune)")

    threshold, acc = calibrate(scored)
    print(f"\nSeuil optimal trouve : {threshold:.4f}  (accuracy sur cet echantillon : {acc:.3f})")
    print(f"-> reporter cette valeur dans RERANKER_REFUSAL_THRESHOLD, refusal_gate.py")
