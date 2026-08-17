#!/usr/bin/env python3
"""
pipeline.py — ORCHESTRATEUR de la phase de requete complete : query processing
-> retrieval (une ou plusieurs sous-questions) -> generation.

Compression de contexte VOLONTAIREMENT non branchee ici pour l'instant
(etape mise de cote ; il suffirait d'un appel compress() avant generate()).

Ne contient AUCUNE logique metier propre -- compose seulement :
  query_processing.process_query()   (reformulation + decomposition)
  retriever.retrieve()               (hybride BM25+dense+rerank, par (sous-)question)
  fusion round-robin                 (entrelacement des hits des sous-questions,
                                       dedup par parent_id, pour preserver des
                                       slots a la question globale dans le top-k)

  generator.generate()               (reponse finale)

C'est ce fichier (pas evaluate.py) que le test ET toute interface interactive
(CLI/Streamlit) doivent appeler -- un seul chemin de code pour les deux, comme
prevu depuis le debut de cette refonte.
"""

from dataclasses import dataclass
from collections.abc import Iterator

from .query_processing import process_query
from .retriever import retrieve
from .refusal_gate import should_refuse_reranker
from .generator import generate, generate_stream, verify_answer, DEFAULT_SYSTEM_PROMPT, REFUSAL_MESSAGE


@dataclass
class RAGResult:
    """Resultat d'un appel `answer()` : question, contexte et reponse produites.

    ``refused=True`` signifie qu'un mecanisme de refus (M1 reranker pre-generation
    ou M2 LLM judge post-generation) a remplace la reponse par ``REFUSAL_MESSAGE``.
    En mode ``stream=True``, ``answer`` est vide et les tokens sont produits par
    ``answer_stream`` (generateur a iterer cote appelant).
    """

    query: str
    rewritten_query: str
    sub_queries: list[str]
    hits: list[dict]        # dicts complets (text/dist/meta) -- pour l'évaluateur (parent_id, source, pages...)
    contexts: list[str]     # juste les textes -- pratique pour la génération / un affichage rapide
    answer: str
    refused: bool           # True si refusal_gate a bloque AVANT generate() (pas d'appel LLM fait)
    answer_stream: Iterator[str] | None = None  # generateur de tokens en mode stream=True


def answer(query: str, k: int = 4, system_prompt: str | None = None,
           use_query_processing: bool = True, use_refusal_gate: bool = True,
           history: str | None = None, stream: bool = False) -> RAGResult:
    """use_query_processing=False : saute process_query() entierement (1 appel LLM
    de moins par question) -- utilise la question brute directement pour le
    retrieval. Sacrifice la reformulation/decomposition ; utile en mode degrade
    (CPU pur, pas de GPU) pour diviser environ par 2 le nombre d'appels LLM par
    question le temps de retrouver du GPU -- pas destine aux chiffres finaux.

    use_refusal_gate=False : desactive la decision de refus explicite (utile pour
    comparer avec/sans lors de la calibration, ou pour retrouver l'ancien
    comportement 100% implicite au prompt).

    history (optionnel) : str pre-formaté contenant l'historique de conversation
    (sortie de ConversationMemory.get_formatted_history()). None = mode Q&A direct.
    Si fourni, il est passe a la fois au query processing (resolution des anaphores
    avant retrieval) et au generator (contexte conversationnel dans le prompt).

    Fusion multi sous-questions : quand la question est decomposee, chaque
    sous-question est retrievee separement puis les hits sont ENTRELACES en
    round-robin (dedup par parent_id) plutot que concatenes : cela preserve des
    slots dans le top-k pour la question globale, meme face a des sous-questions
    plus etroites dont les scores cosinus sont artificiellement plus eleves.
    (Avant : concatener sans re-trier tronquait prematurement les sous-questions
    suivantes et faussait hit@k/MRR a la baisse.)"""
    if use_query_processing:
        proc = process_query(query, history=history)
        # Toujours inclure la question reformulee (intention globale), et AJOUTER
        # les sous-questions en bonus pour ameliorer le rappel sur des notions
        # specifiques. Ne plus jeter la question globale : c'etait la cause de la
        # degradation mesuree au Run 2 (hit@4 -16.7pts, MRR -23.9pts).
        #
        # On deduplique les sous-questions quasi-identiques a la question reformulee
        # (evite le bruit et les appels retrieve() redondants).
        rewritten, sub_queries = proc["rewritten"], proc["sub_queries"]
        queries_to_retrieve = [rewritten]
        seen = {rewritten.strip().lower()}
        for sq in sub_queries:
            key = sq.strip().lower()
            if key not in seen:
                seen.add(key)
                queries_to_retrieve.append(sq)
    else:
        queries_to_retrieve = [query]
        rewritten, sub_queries = query, []

    # --- Retrieval + fusion ---
    if len(queries_to_retrieve) == 1:
        # Cas simple (pas de decomposition) : top-k direct, comportement historique.
        all_hits = retrieve(queries_to_retrieve[0], k=k)
    else:
        # Round-robin : interleave les hits de chaque query (1er de la reformulee,
        # 1er de sub1, 1er de sub2, 2e de la reformulee...) pour garantir que la
        # question globale garde des slots dans le top-k, meme face a des
        # sous-questions plus etroites dont les scores cosinus sont artificiellement
        # plus eleves.
        hits_per_query = [retrieve(q, k=k) for q in queries_to_retrieve]
        all_hits = []
        seen_pids = set()
        max_len = max((len(h) for h in hits_per_query), default=0)
        for i in range(max_len):
            for hits in hits_per_query:
                if i < len(hits):
                    pid = hits[i]["meta"].get("parent_id")
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        all_hits.append(hits[i])
                        if len(all_hits) >= k:
                            break
            if len(all_hits) >= k:
                break

    # --- Mecanisme 1 : refus RERANKER (cross-encoder, zero cout supplementaire) ---
    # Utilise le score du reranker (hits[0]["dist"] en mode hybrid_rerank) pour
    # decider si le contexte est pertinent. Le cross-encoder est un BIEN meilleur
    # juge de pertinence que le score cosine/BM25 — c'est litteralement sa fonction.
    # Ne coute RIEN de plus : le reranker tourne deja dans hybrid_rerank.
    if use_refusal_gate and should_refuse_reranker(all_hits):
        return RAGResult(
            query=query, rewritten_query=rewritten, sub_queries=sub_queries,
            hits=all_hits, contexts=[h["text"] for h in all_hits],
            answer=REFUSAL_MESSAGE, refused=True,
        )

    # --- Mode streaming : bypass M2 (necessite la reponse complete) ---
    if stream:
        return RAGResult(
            query=query, rewritten_query=rewritten, sub_queries=sub_queries,
            hits=all_hits, contexts=[h["text"] for h in all_hits],
            answer="", refused=False,
            # Le caller itere sur .answer_stream pour afficher les tokens
            # et reconstruit la reponse complete en les concatenant.
            answer_stream=generate_stream(
                query, all_hits, system_prompt=system_prompt, history=history
            ),
        )

    answer_text = generate(query, all_hits, system_prompt=system_prompt, history=history)

    # --- Mecanisme 2 : refus POST-generation (LLM Judge, verifie la fidelite) ---
    # Apres generation, un 2e LLM verifie que la reponse est bien ancree dans le
    # contexte. Si la reponse contient des informations hors-contexte (hallucination,
    # memoire parametrique), on la remplace par REFUSAL_MESSAGE.
    #
    # Principe : « Verifier une reponse est plus facile que la generer. »
    # On detecte le cas ou le LLM "triche" en utilisant ses connaissances internes
    # plutot que le contexte fourni — cas confirme au Run 3 (Bug #4).
    if use_refusal_gate and not verify_answer(query, answer_text, all_hits):
        return RAGResult(
            query=query, rewritten_query=rewritten, sub_queries=sub_queries,
            hits=all_hits, contexts=[h["text"] for h in all_hits],
            answer=REFUSAL_MESSAGE, refused=True,
        )

    return RAGResult(
        query=query,
        rewritten_query=rewritten,
        sub_queries=sub_queries,
        hits=all_hits,
        contexts=[h["text"] for h in all_hits],
        answer=answer_text,
        refused=False,
    )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "qu'est-ce qu'une cellule LSTM et comment se compare-t-elle a un GRU ?"
    res = answer(q)
    print(f"Question           : {res.query}")
    print(f"Reformulee          : {res.rewritten_query}")
    print(f"Sous-questions      : {res.sub_queries}")
    print(f"Nb contextes retenus: {len(res.contexts)}")
    print(f"\nReponse :\n{res.answer}")