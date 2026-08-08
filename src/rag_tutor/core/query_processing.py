#!/usr/bin/env python3
"""
query_processing.py — SEULE RESPONSABILITE : reformuler la question de
l'etudiant pour plus de clarte, et la decomposer en sous-questions si elle
porte sur plusieurs notions distinctes -- pour ameliorer le RAPPEL du
retrieval (chaque notion est cherchee separement plutot que noyee dans une
requete composite).

UN SEUL appel LLM (via llm_client.py) fait reformulation + decomposition en
meme temps -- pas deux appels separes : plus rapide sur un Ollama partage/Colab
deja lent. Le LLM repond en JSON ; en cas d'echec de parsing (LLM bavard, JSON
invalide, timeout...) on retombe SANS exception sur la question d'origine, non
decomposee -- degradation silencieuse cote qualite, jamais de crash du pipeline.

Ne fait NI retrieval NI generation -- produit seulement les requetes que
retriever_hybride.retrieve() utilisera en aval.

API publique :
  process_query(query) -> {"rewritten": str, "sub_queries": list[str]}
    sub_queries est vide si la question est deja simple/atomique.
"""

import json
import re

from .llm_client import chat

_SYSTEM_PROMPT = (
    "Tu reformules et, si necessaire, decomposes une question d'etudiant pour "
    "ameliorer une recherche documentaire dans un cours. Renvoie UNIQUEMENT un "
    "objet JSON, rien d'autre (pas de texte avant/apres, pas de balises "
    "markdown), avec exactement ces deux cles :\n"
    '  "rewritten" : la question reformulee -- plus claire, complete, sans '
    "ambiguite ni faute -- en francais, en conservant l'intention exacte de "
    "l'etudiant.\n"
    '  "sub_queries" : liste de sous-questions INDEPENDANTES si la question '
    "porte sur PLUSIEURS notions radicalement differentes et eloignees dans le "
    "cours ; liste VIDE si la question est deja simple/atomique.\n"
    "IMPORTANT :\n"
    "- Limite-toi a 2 ou 3 sous-questions MAXIMUM.\n"
    "- Ne decompose PAS si la question peut etre repondue par un seul passage du cours.\n"
    "- Chaque sous-question doit etre autocontenue (comprehensible sans la question d'origine).\n"
    "- Ne decompose jamais une question deja simple juste pour en faire plusieurs."
)


def _parse(raw, original_query):
    """Extrait le JSON de la reponse LLM ; repli sur la question d'origine si
    le parsing echoue ou si la forme ne correspond pas a ce qui est attendu."""
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            rewritten = data.get("rewritten") or original_query
            sub_queries = data.get("sub_queries") or []
            if isinstance(rewritten, str) and isinstance(sub_queries, list) \
                    and all(isinstance(s, str) for s in sub_queries):
                return {"rewritten": rewritten, "sub_queries": sub_queries}
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"rewritten": original_query, "sub_queries": []}


def process_query(query, model=None):
    """Reformule `query` et la decompose si besoin. Ne leve jamais d'exception
    liee au LLM/parsing -- repli sur la question d'origine en cas de probleme."""
    kwargs = {"model": model} if model else {}
    try:
        raw = chat(_SYSTEM_PROMPT, query, **kwargs)
    except Exception:
        return {"rewritten": query, "sub_queries": []}
    return _parse(raw, query)