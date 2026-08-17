#!/usr/bin/env python3
"""
refusal_gate.py — MECANISMES DE REFUS :

  1) RERANKER-based (pre-generation, zero cout supplementaire) :
     should_refuse_reranker(hits) → bool
     Utilise le score DU RERANKER (cross-encoder bge-reranker-v2-m3) pour
     decider si le contexte est pertinent. Le reranker est un bien meilleur
     juge de pertinence que le score cosine/BM25 — c'est litteralement sa
     fonction. Ne coute RIEN de plus puisqu'il tourne deja en hybrid_rerank.

  2) POST-generation (LLM judge) : verify_answer(question, answer, hits) → bool
     (defini dans generator.py) — un 2e LLM (qwen3:8b) verifie que la reponse
     est bien ancree dans le contexte fourni. Decide APRES la generation,
     plus fiable qu'un simple score de retrieval.

  NOTE : l'ancien mecanisme score-based (should_refuse, REFUSAL_THRESHOLD) a
  ete supprime (commit ~2026-08-06). Le score du meilleur hit de retrieval
  (RRF ou cosine) est un signal trop faible pour discriminer answerable /
  unanswerable sur ce dataset — le seuil etait au minimum theorique RRF et
  ne declenchait jamais. Le reranker le remplace avantageusement.

API publique :
  should_refuse_reranker(hits, threshold=RERANKER_REFUSAL_THRESHOLD) -> bool
  calibrate(scored_examples) -> tuple[float, float]   # (seuil optimal, accuracy)
"""

# =====================================================
# MECANISME 1 : score du reranker (pre-generation, cross-encoder)
# =====================================================

# Seuil sur le score du cross-encoder bge-reranker-v2-m3 en-dessous duquel on
# refuse de repondre. Le reranker donne des logits (pas des probabilites) —
# l'echelle approximative est [-10, +10], les scores > 0 indiquent une pertinence
# positive. Valeur CALIBREE via evaluation/calibrate.py sur eval/test_set_v2.json
# (mode hybrid_rerank) — cf. eval/reranker_calibration.json.
RERANKER_REFUSAL_THRESHOLD = 0.1119


def should_refuse_reranker(hits: list[dict], threshold: float = RERANKER_REFUSAL_THRESHOLD) -> bool:
    """Refuse si le score du meilleur hit (score du reranker en mode hybrid_rerank,
    ou score RRF/cosine sinon) est sous le seuil. Ne coute rien : le reranker
    tourne deja dans hybrid_rerank, son score est dans hits[0]["dist"].

    En mode hybrid_rerank : hits[0]["dist"] = score du cross-encoder (logit).
    En mode hybrid : hits[0]["dist"] = score RRF (peu discriminant).
    En mode dense : hits[0]["dist"] = similarite cosinus.

    Retourne True si la reponse doit ETRE REFUSEE (score < seuil)."""
    if not hits:
        return True
    top_score = hits[0].get("dist")
    if top_score is None:
        return False
    return top_score < threshold


# =====================================================
# CALIBRATION (generique — pour les trois mecanismes)
# =====================================================

def calibrate(scored_examples: list[tuple[float, bool]]) -> tuple[float, float]:
    """scored_examples : liste de (top_score, is_unanswerable: bool). Balaie les
    seuils possibles (chaque score observe) et renvoie celui qui maximise
    l'accuracy refus/reponse sur cet echantillon -- point de depart, pas
    scientifiquement optimal (pas de validation croisee), mais largement
    suffisant pour un seuil de decision simple comme celui-ci."""
    if not scored_examples:
        raise ValueError("scored_examples est vide -- rien a calibrer")

    candidates = sorted(set(s for s, _ in scored_examples))
    best_threshold, best_acc = candidates[0], -1.0
    for t in candidates:
        correct = sum(
            1 for score, is_unanswerable in scored_examples
            if (score < t) == is_unanswerable
        )
        acc = correct / len(scored_examples)
        if acc > best_acc:
            best_threshold, best_acc = t, acc
    return best_threshold, best_acc
