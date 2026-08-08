#!/usr/bin/env python3
"""
conversation_memory.py — Gestion de l'historique de conversation pour le tuteur
pédagogique conversationnel v2.

Responsabilités :
  1. Stocker les tours de conversation (student / tutor)
  2. Détecter un changement de sujet → réinitialiser le contexte retrieval
  3. Compresser les anciens messages quand on approche de la limite de contexte
  4. Formater l'historique pour inclusion dans le prompt de génération

API publique :
  ConversationMemory.add_turn(role, content) -> None
  ConversationMemory.get_formatted_history(max_tokens=4000) -> str
  ConversationMemory.is_new_topic(query, threshold=0.3) -> bool
  ConversationMemory.clear() -> None
  ConversationMemory.last_n_turns(n) -> list
"""

import re
from collections import deque


class ConversationMemory:
    """Mémoire de conversation avec compression automatique.

    Stratégie :
      - Stocke les N derniers tours intacts (fenêtre récente, défaut 6 tours)
      - Les tours plus anciens sont compressés en un résumé (via le LLM si
        disponible, sinon tronqués)
      - Détection de changement de sujet : compare les mots-clés de la
        nouvelle question avec ceux des 3 derniers tours
    """

    def __init__(self, recent_window=6, max_summary_tokens=500):
        self._turns: list[dict] = []        # [{"role": "student"|"tutor", "content": str}]
        self._summary: str = ""             # résumé compressé des tours anciens
        self._recent_window = recent_window
        self._max_summary_tokens = max_summary_tokens

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str):
        """Ajoute un tour à l'historique. role ∈ {student, tutor}."""
        if role not in ("student", "tutor"):
            raise ValueError(f"role doit être 'student' ou 'tutor', reçu: {role}")
        self._turns.append({"role": role, "content": content})

    def get_formatted_history(self, max_tokens: int = 4000) -> str:
        """Retourne l'historique formaté pour inclusion dans le prompt.

        Structure :
          [RÉSUMÉ DE LA CONVERSATION PRÉCÉDENTE]
          ... (si disponible)

          [DERNIERS ÉCHANGES]
          Étudiant : ...
          Tuteur : ...
        """
        parts = []

        if self._summary:
            parts.append(f"[RÉSUMÉ DE LA CONVERSATION PRÉCÉDENTE]\n{self._summary}\n")

        recent = self._turns[-self._recent_window:] if self._recent_window > 0 else self._turns
        if recent:
            parts.append("[DERNIERS ÉCHANGES]")
            for turn in recent:
                role_label = "Étudiant" if turn["role"] == "student" else "Tuteur"
                parts.append(f"{role_label} : {turn['content']}")

        history_text = "\n".join(parts)

        # Truncation grossière si trop long (on coupe les tours les plus anciens)
        if self._estimate_tokens(history_text) > max_tokens:
            history_text = self._truncate_history(recent, self._summary, max_tokens)

        return history_text

    def is_new_topic(self, query: str, threshold: float = 0.3) -> bool:
        """Détecte si la question change radicalement de sujet.

        Compare les mots-clés (mots >= 4 lettres) de la question avec ceux
        des 3 derniers tours. Si le chevauchement est < threshold → nouveau sujet.

        Retourne True si le sujet a changé, False sinon.
        """
        if len(self._turns) < 2:
            return False

        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return False

        # Prendre les 3 derniers tours (student + tutor)
        recent_turns = self._turns[-3:]
        recent_text = " ".join(t["content"] for t in recent_turns)
        recent_keywords = self._extract_keywords(recent_text)

        if not recent_keywords:
            return False

        overlap = len(query_keywords & recent_keywords) / len(query_keywords)
        return overlap < threshold

    def clear(self):
        """Réinitialise la mémoire."""
        self._turns.clear()
        self._summary = ""

    def last_n_turns(self, n: int = 1) -> list:
        """Retourne les N derniers tours (le plus récent en dernier)."""
        return self._turns[-n:] if n > 0 else []

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def summary(self) -> str:
        return self._summary

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, llm_summarize_fn=None):
        """Compresse l'historique : résume les tours anciens, garde les
        recent_window derniers intacts.

        Si llm_summarize_fn est fourni (callable prenant un texte et retournant
        un résumé), il est utilisé pour le résumé. Sinon, compression par
        troncature simple.
        """
        if len(self._turns) <= self._recent_window:
            return  # pas assez de tours pour justifier une compression

        old_turns = self._turns[:-self._recent_window]
        old_text = "\n".join(
            f"{'Étudiant' if t['role'] == 'student' else 'Tuteur'} : {t['content']}"
            for t in old_turns
        )

        if llm_summarize_fn:
            try:
                self._summary = llm_summarize_fn(
                    f"Résume cette conversation en 2-3 phrases maximum, en "
                    f"gardant les sujets abordés et les points clés :\n\n{old_text}"
                )
            except Exception:
                self._summary = self._truncate_summary(old_text)
        else:
            self._summary = self._truncate_summary(old_text)

        # Remplacer les anciens tours par le résumé (on garde juste le résumé)
        # Note : on ne supprime pas les tours, le résumé est stocké à part
        # et get_formatted_history() l'inclut automatiquement

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extrait les mots-clés (mots >= 4 lettres, hors stopwords basiques)."""
        stopwords = {
            "cette", "cette", "dans", "pour", "avec", "plus", "moins",
            "tout", "très", "être", "avoir", "faire", "peut", "aussi",
            "alors", "comme", "entre", "deux", "leur", "dont", "quelle",
            "qu'est", "comment", "pourquoi", "différence", "the", "and",
            "what", "that", "this", "from", "with", "does", "between",
            "explique", "expliquer",
        }
        words = re.findall(r"\w{4,}", text.lower())
        return {w for w in words if w not in stopwords}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimation grossière : ~1.3 token par mot (français/anglais mix)."""
        return int(len(text.split()) * 1.3)

    def _truncate_summary(self, text: str) -> str:
        """Tronque un texte en gardant les premières phrases."""
        words = text.split()
        if len(words) <= self._max_summary_tokens:
            return text
        return " ".join(words[:self._max_summary_tokens]) + "..."

    def _truncate_history(self, recent_turns: list, summary: str, max_tokens: int) -> str:
        """Tronque l'historique pour respecter max_tokens, en sacrifiant
        les tours les plus anciens en premier."""
        parts = []
        if summary:
            parts.append(f"[RÉSUMÉ]\n{summary}")

        # On garde au minimum les 2 derniers tours
        min_turns = min(2, len(recent_turns))
        kept = list(recent_turns)
        while kept and self._estimate_tokens(
            "\n".join(parts + [f"{t['role']}: {t['content']}" for t in kept])
        ) > max_tokens and len(kept) > min_turns:
            kept.pop(0)  # supprimer le plus ancien

        parts.append("[DERNIERS ÉCHANGES]")
        for t in kept:
            role_label = "Étudiant" if t["role"] == "student" else "Tuteur"
            parts.append(f"{role_label} : {t['content']}")

        return "\n".join(parts)


# =====================================================
# Helpers pratiques
# =====================================================

def summarize_with_llm(text: str, llm_client) -> str:
    """Résume un texte via le LLM (utilise le client Ollama générique).

    llm_client doit avoir une fonction chat(system_prompt, user_message) -> str.
    """
    from ..core.llm_client import chat
    return chat(
        "Tu es un assistant qui résume des conversations. "
        "Réponds en 2-3 phrases maximum, en français.",
        text,
        temperature=0,
    )
