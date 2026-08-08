#!/usr/bin/env python3
"""
refusal_gate.py — MECANISMES DE REFUS :

  1) RERANKER-based (pre-generation, zero cout supplementaire) :
     should_refuse_reranker(hits) → bool
     Utilise le score DU RERANKER (cross-encoder bge-reranker-v2-m3) pour
     decider si le contexte est pertinent. Le reranker est un bien meilleur
     juge de pertinence que le score cosine/BM25 — c'est litteralement sa
     fonction. Ne coute RIEN de plus puisqu'il tourne deja en hybrid_rerank.

  2) POST-generation (confidence-based) : check_confidence(answer_text) → bool
     Parse le score [CONFIANCE: X/5] que le generateur integre en tete de
     reponse (cf. generator.py, DEFAULT_SYSTEM_PROMPT). Decide APRES la
     generation — plus fiable car le LLM 14B evalue lui-meme si le contexte
     est suffisant, ce qu'un simple score de retrieval ne peut pas faire.

  NOTE : l'ancien mecanisme score-based (should_refuse, REFUSAL_THRESHOLD) a
  ete supprime (commit ~2026-08-06). Le score du meilleur hit de retrieval
  (RRF ou cosine) est un signal trop faible pour discriminer answerable /
  unanswerable sur ce dataset — le seuil etait au minimum theorique RRF et
  ne declenchait jamais. Le reranker le remplace avantageusement.

API publique :
  should_refuse_reranker(hits, threshold=RERANKER_REFUSAL_THRESHOLD) -> bool
  parse_confidence(answer_text) -> int 1-5 | None
  check_confidence(answer_text, threshold=CONFIDENCE_THRESHOLD) -> bool
  calibrate(scored_examples) -> float
"""

import re


# =====================================================
# MECANISME 2 : score de confiance du LLM (post-generation)
# =====================================================

# Seuil de confiance en-dessous duquel on refuse la reponse (remplacee par
# REFUSAL_MESSAGE).
#
# En two-pass (check_confidence_llm), le LLM juge est naturellement conservateur
# → un seuil de 2 est plus approprie qu'un seuil de 3. Refuser uniquement si :
#   - NON explicite (le LLM juge que le contexte ne permet pas de repondre)
#   - confiance = 1 (aucune information pertinente)
#   - confiance = 2 (mention vague, insuffisante)
#
# Ce seuil DOIT etre calibre empiriquement (run complet + analyse).
CONFIDENCE_THRESHOLD = 2

# Pattern pour extraire [CONFIANCE: X/5] en debut de reponse.
# Robuste aux variations : espaces, majuscules, X/5 ou X sur 5.
_CONFIDENCE_RE = re.compile(
    r'^\s*\[?\s*CONFIANCE\s*:\s*(\d)\s*(?:/\s*5|\s*sur\s*5)?\s*\]?\s*$',
    re.MULTILINE | re.IGNORECASE
)


def parse_confidence(answer_text):
    """Extrait le score de confiance (int 1-5) d'une reponse du generateur.
    Cherche [CONFIANCE: X/5] uniquement en TOUT DEBUT de reponse (premiere
    ligne ou premier bloc). Retourne None si le format est absent ou invalide
    → le pipeline ne refuse PAS par defaut (conservateur : en cas de doute,
    on fait confiance a la reponse).

    Robuste au markdown : ** et __ sont neutralises avant parsing (les LLMs
    ont tendance a wrapper le tag en bold)."""
    if not answer_text:
        return None
    # Neutraliser le markdown bold (** / __) que le LLM pourrait ajouter
    # autour du tag — pattern ultra-courant avec les modeles instruits.
    cleaned = answer_text[:200].replace('**', '').replace('__', '')
    m = _CONFIDENCE_RE.search(cleaned)
    if not m:
        return None
    score = int(m.group(1))
    if 1 <= score <= 5:
        return score
    return None


def check_confidence(answer_text, threshold=CONFIDENCE_THRESHOLD):
    """Verifie si la confiance du LLM dans sa reponse est suffisante.
    Retourne True si la reponse doit ETRE REFUSEE (confiance < seuil).
    Retourne False si la reponse est acceptable (confiance >= seuil ou
    format non parse → conservateur)."""
    score = parse_confidence(answer_text)
    if score is None:
        return False   # format absent → ne pas bloquer (conservateur)
    return score < threshold


# =====================================================
# MECANISME 1 : score du reranker (pre-generation, cross-encoder)
# =====================================================

# Seuil sur le score du cross-encoder bge-reranker-v2-m3 en-dessous duquel on
# refuse de repondre. Le reranker donne des logits (pas des probabilites) —
# l'echelle approximative est [-10, +10], les scores > 0 indiquent une pertinence
# positive. Ce seuil DOIT etre calibre via calibrate_reranker_refusal.py.
#
# -5.0 = valeur volontairement basse (placeholder) : ne declenche quasi jamais
# en attendant la calibration. A ajuster apres avoir lance le script de calibration.
RERANKER_REFUSAL_THRESHOLD = 0.1119


def should_refuse_reranker(hits, threshold=RERANKER_REFUSAL_THRESHOLD):
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

def calibrate(scored_examples):
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
