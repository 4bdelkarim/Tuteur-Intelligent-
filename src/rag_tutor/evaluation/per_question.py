#!/usr/bin/env python3
"""
per_question.py — Évaluation en temps réel d'une réponse du pipeline RAG.

Contrairement à evaluate.py (évaluation globale sur un golden dataset avec
références), ce module fournit des métriques PAR QUESTION, calculables sans
golden dataset :

  Niveau 1 — Stats internes du pipeline (coût nul, déjà calculées)
    • requête reformulée, sous-questions
    • nb de contextes, nb de parents uniques
    • scores du reranker (top, moyenne, range)
    • statut du refusal gate (M1/M2)

  Niveau 2 — Juge LLM (1 appel LLM supplémentaire, ~5-15s)
    • Fidélité : vérifie que la réponse est ancrée dans le contexte
      (ANCRÉ / HORS-CONTEXTE), via verify_answer() de generator.py

  Niveau 3 — Scores retrieval simulés (coût nul, heuristiques lexicales)
    • Overlap lexical question ↔ contextes
    • Diversité des sources (pdf/web)
    • Couverture sémantique estimée

API publique :
  evaluate_response(query, result, run_judge=True, run_retrieval_scores=True) -> dict
  format_eval_report(report) -> str
"""

import re
from ..core.generator import verify_answer


def evaluate_response(query, result, answer_text=None, run_judge=True, run_retrieval_scores=True):
    """Évalue une réponse du pipeline et retourne un dictionnaire structuré.

    Paramètres
    ----------
    query : str
        La question originale posée par l'utilisateur.
    result : RAGResult
        Le résultat retourné par pipeline.answer().
    answer_text : str, optionnel
        Texte complet de la réponse. Si None, utilise result.answer.
        Nécessaire en mode streaming où result.answer est vide.
    run_judge : bool
        Si True, lance le juge LLM (verify_answer) pour évaluer la fidélité.
        Ajoute ~5-15s par question.
    run_retrieval_scores : bool
        Si True, calcule des scores de retrieval heuristiques (overlap lexical,
        diversité des sources). Coût nul.

    Retourne
    --------
    dict avec les clés : "pipeline", "judge", "retrieval"
    """
    report = {}

    # ==================================================================
    # NIVEAU 1 : Stats internes du pipeline (déjà dans RAGResult)
    # ==================================================================
    hits = result.hits if result.hits else []
    contexts = result.contexts if result.contexts else []
    full_answer = answer_text if answer_text is not None else (result.answer or "")

    pipeline_stats = {
        "rewritten_query": result.rewritten_query,
        "sub_queries": result.sub_queries,
        "nb_sub_queries": len(result.sub_queries) if result.sub_queries else 0,
        "nb_contexts": len(contexts),
        # Parents uniques (déduplication par parent_id)
        "nb_unique_parents": len(set(
            h.get("meta", {}).get("parent_id")
            for h in hits
            if h.get("meta", {}).get("parent_id") is not None
        )),
        "refused": result.refused,
        "answer_length": len(full_answer) if full_answer else 0,
    }

    # Scores de retrieval (disponibles dans hits[].dist)
    scores = [h.get("dist") for h in hits if h.get("dist") is not None]
    if scores:
        pipeline_stats["reranker_score_top"] = round(scores[0], 3)
        pipeline_stats["reranker_score_avg"] = round(sum(scores) / len(scores), 3)
        pipeline_stats["reranker_score_min"] = round(min(scores), 3)
        pipeline_stats["reranker_score_max"] = round(max(scores), 3)
        pipeline_stats["reranker_score_range"] = round(max(scores) - min(scores), 3) if len(scores) > 1 else 0

    # Diagnostic refusal gate : quel mécanisme a déclenché ?
    from ..core.refusal_gate import should_refuse_reranker
    if result.refused:
        m1_triggered = should_refuse_reranker(hits)
        pipeline_stats["refusal_m1_reranker"] = m1_triggered
        pipeline_stats["refusal_m2_verify"] = not m1_triggered  # si pas M1, c'est M2
    else:
        pipeline_stats["refusal_m1_reranker"] = False
        pipeline_stats["refusal_m2_verify"] = False

    report["pipeline"] = pipeline_stats

    # ==================================================================
    # NIVEAU 2 : Juge LLM (vérification post-generation de la fidélité)
    # ==================================================================
    if run_judge and not result.refused and full_answer:
        try:
            is_anchored = verify_answer(query, full_answer, hits)
            report["judge"] = {
                "faithfulness": "ANCRÉ" if is_anchored else "HORS-CONTEXTE",
                "is_anchored": is_anchored,
            }
        except Exception as e:
            report["judge"] = {
                "faithfulness": "ERREUR",
                "error": str(e)[:200],
            }
    elif result.refused:
        report["judge"] = {
            "faithfulness": "N/A (refusé)",
            "is_anchored": None,
        }
    else:
        report["judge"] = None

    # ==================================================================
    # NIVEAU 3 : Scores retrieval simulés (heuristiques lexicales)
    # ==================================================================
    if run_retrieval_scores and contexts:
        retrieval_stats = _compute_retrieval_stats(query, hits, contexts)
        report["retrieval"] = retrieval_stats
    else:
        report["retrieval"] = None

    # ==================================================================
    # NIVEAU 4 : Citations — quels documents récupérés ont été cités ?
    # ==================================================================
    if full_answer and hits:
        report["citations"] = _analyze_citations(full_answer, hits)
    else:
        report["citations"] = None

    return report


# ======================================================================
# Helpers privés — Niveau 3
# ======================================================================

def _extract_keywords(text: str, min_len: int = 3) -> set:
    """Extrait les mots-clés (mots >= min_len lettres, filtrés stopwords).
    min_len=3 pour capturer les acronymes techniques (CNN, RNN, GPU...)."""
    # stopwords >= 3 lettres (min_len=3 capture les acronymes techniques)
    stopwords = {
        # français
        "cette", "dans", "pour", "avec", "plus", "moins", "tout", "très",
        "être", "avoir", "faire", "peut", "aussi", "alors", "comme", "entre",
        "deux", "leur", "dont", "quelle", "qu'est", "comment", "pourquoi",
        "différence", "explique", "expliquer", "quels", "quelles",
        "est-ce", "sont", "sont-ils", "peut-on",
        "les", "des", "une", "est", "pas", "que", "qui", "sur",
        "par", "son", "ses", "aux", "ces", "très", "c'est",
        # anglais
        "the", "and", "what", "that", "this", "from", "with",
        "does", "between", "are", "not", "for", "can", "has",
        "its", "use", "how", "was", "but", "all", "any",
    }
    words = re.findall(rf"\w{{{min_len},}}", text.lower())
    return {w for w in words if w not in stopwords}


def _analyze_citations(answer_text: str, hits: list) -> dict:
    """Analyse quels documents récupérés ont été cités dans la réponse.

    Parse les citations [DOC X] dans la réponse et vérifie combien de
    documents récupérés ont effectivement été utilisés.

    Retourne un dict avec :
      - cited_docs : liste des numéros de docs cités
      - nb_cited : nombre de docs distincts cités
      - nb_retrieved : nombre de docs récupérés
      - citation_rate : fraction des docs récupérés qui ont été cités
      - uncited_docs : liste des docs récupérés mais non cités
    """
    # Extraire les numéros de documents cités : [DOC 1], [DOC 2], etc.
    cited = set()
    for m in re.finditer(r'\[DOC\s*(\d+)\]', answer_text, re.IGNORECASE):
        cited.add(int(m.group(1)))

    nb_retrieved = len(hits)
    nb_cited = len(cited)
    citation_rate = nb_cited / nb_retrieved if nb_retrieved > 0 else 0.0

    # Docs récupérés mais non cités
    uncited = [i + 1 for i in range(nb_retrieved) if (i + 1) not in cited]

    # Pour chaque doc cité, extraire la source
    cited_sources = {}
    for doc_num in sorted(cited):
        if doc_num <= len(hits):
            meta = hits[doc_num - 1].get("meta", {})
            src = meta.get("source", "?")
            cited_sources[doc_num] = _trunc(src, 30)

    return {
        "cited_docs": sorted(cited),
        "nb_cited": nb_cited,
        "nb_retrieved": nb_retrieved,
        "citation_rate": round(citation_rate, 2),
        "uncited_docs": uncited,
        "cited_sources": cited_sources,
    }


def _compute_retrieval_stats(query: str, hits: list, contexts: list) -> dict:
    """Calcule des métriques de retrieval heuristiques (sans golden dataset)."""
    query_keywords = _extract_keywords(query)

    # 1) Overlap lexical question ↔ contextes récupérés
    all_context_text = " ".join(contexts).lower()
    ctx_words = set(re.findall(r"\w{3,}", all_context_text))

    if query_keywords:
        overlap_ratio = len(query_keywords & ctx_words) / len(query_keywords)
    else:
        overlap_ratio = 0.0

    # 2) Diversité des sources (résumé par type)
    source_types = {}
    for h in hits:
        meta = h.get("meta", {}) if isinstance(h, dict) else {}
        src_type = meta.get("source_type", "inconnu")
        source_types[src_type] = source_types.get(src_type, 0) + 1
    source_summary = ", ".join(f"{t} ×{c}" for t, c in sorted(source_types.items()))

    # 3) Sections couvertes
    sections = set()
    for h in hits:
        meta = h.get("meta", {}) if isinstance(h, dict) else {}
        sec = meta.get("section", "")
        if sec:
            sections.add(sec)

    # 4) Longueur totale des contextes
    total_context_chars = sum(len(ctx) for ctx in contexts)

    return {
        "lexical_overlap": round(overlap_ratio, 3),
        "query_keywords": len(query_keywords),
        "matched_keywords": len(query_keywords & ctx_words),
        "source_summary": source_summary,
        "nb_unique_sections": len(sections),
        "total_context_chars": total_context_chars,
    }


# ======================================================================
# Formatage pour affichage dans le CLI
# ======================================================================

def format_eval_report(report: dict) -> str:
    """Formate un rapport d'évaluation en texte lisible pour le CLI.

    Retourne une str multilingue prête à afficher.
    """
    lines = []
    sep = "─" * 62

    lines.append(f"┌─{sep}┐")
    lines.append(f"│ {'📊 ÉVALUATION PAR QUESTION':<62} │")

    # --- Pipeline ---
    p = report.get("pipeline", {})
    if p:
        lines.append(f"│ {'─' * 62} │")
        lines.append(f"│ {'🔧 PIPELINE':<62} │")
        lines.append(_fmt_line(f"  Requête reformulée", _trunc(p.get("rewritten_query", ""), 48)))
        sq = p.get("sub_queries", [])
        if sq:
            lines.append(_fmt_line(f"  Sous-questions", str(p.get("nb_sub_queries", 0))))
            for i, sq_item in enumerate(sq[:3]):
                lines.append(_fmt_line(f"    [{i+1}]", _trunc(sq_item, 47)))
        lines.append(_fmt_line(f"  Contextes récupérés",
                                f"{p.get('nb_contexts', 0)} ({p.get('nb_unique_parents', 0)} parents uniques)"))
        if "reranker_score_top" in p:
            lines.append(_fmt_line(f"  Score reranker",
                                    f"top={p['reranker_score_top']:.3f}  "
                                    f"avg={p['reranker_score_avg']:.3f}  "
                                    f"range=[{p['reranker_score_min']:.3f}, {p['reranker_score_max']:.3f}]"))
        # Refusal gate
        refused = p.get("refused", False)
        if refused:
            m1 = "M1:✓" if p.get("refusal_m1_reranker") else "M1:✗"
            m2 = "M2:✓" if p.get("refusal_m2_verify") else "M2:✗"
            lines.append(_fmt_line(f"  Refus", f"OUI ({m1}, {m2})"))
        else:
            lines.append(_fmt_line(f"  Refus", "non"))
        lines.append(_fmt_line(f"  Longueur réponse", f"{p.get('answer_length', 0)} car."))

    # --- Juge LLM ---
    j = report.get("judge")
    if j:
        lines.append(f"│ {'─' * 62} │")
        lines.append(f"│ {'⚖️  JUGE LLM (qwen3:8b)':<62} │")
        faith = j.get("faithfulness", "?")
        if faith == "ANCRÉ":
            lines.append(_fmt_line(f"  Fidélité", "ANCRÉ ✓"))
        elif faith == "HORS-CONTEXTE":
            lines.append(_fmt_line(f"  Fidélité", "HORS-CONTEXTE ⚠"))
        elif faith and faith.startswith("N/A"):
            lines.append(_fmt_line(f"  Fidélité", faith))
        else:
            lines.append(_fmt_line(f"  Fidélité", str(faith)))

    # --- Retrieval ---
    r = report.get("retrieval")
    if r:
        lines.append(f"│ {'─' * 62} │")
        lines.append(f"│ {'🔍 QUALITÉ RETRIEVAL (lexicale)':<62} │")
        lines.append(_fmt_line(f"  Overlap question↔ctx",
                                f"{r['lexical_overlap']:.0%}  ({r.get('matched_keywords', 0)}/"
                                f"{r.get('query_keywords', 0)} mots-clés)"))
        lines.append(_fmt_line(f"  Sources",
                                f"{r.get('source_summary', '?')} ({r.get('nb_unique_sections', 0)} sections)"))
        lines.append(_fmt_line(f"  Volume contextes", f"{r.get('total_context_chars', 0):,} car."))

    # --- Citations ---
    c = report.get("citations")
    if c:
        lines.append(f"│ {'─' * 62} │")
        lines.append(f"│ {'📖 CITATIONS':<62} │")
        cited_list = ", ".join(f"DOC {d}" for d in c.get("cited_docs", []))
        if cited_list:
            lines.append(_fmt_line(f"  Docs cités", cited_list))
            # Afficher la source pour chaque doc cité
            for doc_num, src in c.get("cited_sources", {}).items():
                lines.append(_fmt_line(f"    [DOC {doc_num}]", src))
        else:
            lines.append(_fmt_line(f"  Docs cités", "AUCUN ⚠"))
        uncited = c.get("uncited_docs", [])
        if uncited:
            uncited_list = ", ".join(f"DOC {d}" for d in uncited)
            lines.append(_fmt_line(f"  Docs non cités", uncited_list))
        lines.append(_fmt_line(f"  Taux de citation",
                                f"{c.get('nb_cited', 0)}/{c.get('nb_retrieved', 0)} = {c.get('citation_rate', 0):.0%}"))

    lines.append(f"└─{sep}┘")
    return "\n".join(lines)


def _fmt_line(label: str, value: str) -> str:
    """Formate une ligne label : value dans la largeur dispo (62)."""
    content = f"{label} : {value}"
    # Tronquer si trop long
    if len(content) > 62:
        content = content[:59] + "..."
    return f"│ {content:<62} │"


def _trunc(text: str, max_len: int) -> str:
    """Tronque un texte à max_len caractères avec ellipsis."""
    if not text:
        return "(vide)"
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
