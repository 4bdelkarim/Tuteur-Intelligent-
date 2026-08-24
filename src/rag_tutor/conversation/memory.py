#!/usr/bin/env python3
"""
memory.py — Gestion de l'historique de conversation pour le tuteur
pédagogique conversationnel v2.

Responsabilités :
  1. Stocker les tours de conversation (student / tutor)
  2. Compresser les anciens messages quand on approche de la limite de contexte
  3. Formater l'historique pour inclusion dans le prompt de génération

API publique :
  ConversationMemory.add_turn(role, content) -> None
  ConversationMemory.get_formatted_history(max_tokens=4000) -> str
  ConversationMemory.clear() -> None

Stratégie de compression (v2) :
  La compression est déclenchée tous les recent_window × 2 tours (défaut : 12).
  Chaque cycle résume exactement recent_window tours en UN SEUL appel LLM :

    reçoit  → {ancien résumé} + {nouveaux 6 tours}
    produit → {nouveau résumé cumulatif}

  Le résumé est donc CUMULATIF : il intègre toute la conversation depuis le
  début, mais via une chaîne de résumés fusionnés plutôt qu'en relisant
  l'intégralité des anciens tours.  Après compression, seuls les
  recent_window derniers tours sont conservés en mémoire (élagage).

  Cette approche garantit :
    - mémoire bornée à recent_window × 2 tours (12 max pour window=6)
    - appels LLM bornés à 1 toutes les recent_window questions
    - texte soumis au LLM de résumé de taille constante (~6 tours + résumé)
    - résumé cumulatif qui préserve l'historique complet
"""

from collections.abc import Callable


class ConversationMemory:
    """Mémoire de conversation avec compression automatique par résumé cumulatif.

    Stratégie :
      - Stocke les N derniers tours intacts (fenêtre récente, défaut 6 tours)
      - Tous les recent_window tours, les plus anciens sont résumés et FUSIONNÉS
        avec le résumé précédent en un seul appel LLM
      - Après compression, les tours résumés sont ÉLAGUÉS (contrairement à la
        v1 qui les conservait indéfiniment, causant une fuite mémoire)
    """

    # Seuil de déclenchement : on ne compresse que quand le nombre de tours
    # atteint recent_window × _COMPRESS_THRESHOLD.
    #   window=6, threshold=2 → compression à 12, 18, 24, ... tours.
    _COMPRESS_THRESHOLD = 2

    def __init__(self, recent_window: int = 6, max_summary_tokens: int = 500) -> None:
        self._turns: list[dict] = []        # [{"role": "student"|"tutor", "content": str}]
        self._summary: str = ""             # résumé cumulatif de TOUS les anciens tours
        self._recent_window = recent_window
        self._max_summary_tokens = max_summary_tokens

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Ajoute un tour à l'historique.  *role* ∈ {student, tutor}."""
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

        # Troncature grossière si trop long (on coupe les tours les plus anciens)
        if self._estimate_tokens(history_text) > max_tokens:
            history_text = self._truncate_history(recent, self._summary, max_tokens)

        return history_text

    def clear(self) -> None:
        """Réinitialise la mémoire (tours + résumé)."""
        self._turns.clear()
        self._summary = ""

    @property
    def turn_count(self) -> int:
        """Nombre de tours actuellement en mémoire (fenêtre récente uniquement)."""
        return len(self._turns)

    @property
    def summary(self) -> str:
        """Résumé cumulatif de tous les tours antérieurs à la fenêtre récente."""
        return self._summary

    # ------------------------------------------------------------------
    # Compression (v2 — résumé cumulatif avec élagage)
    # ------------------------------------------------------------------

    def compress(self, llm_summarize_fn: Callable[[str], str] | None = None) -> None:
        """Compresse les tours les plus anciens en UN SEUL appel LLM.

        Principe :
          Un cycle de compression résume *recent_window* tours et FUSIONNE
          ce nouveau résumé avec le résumé cumulatif précédent.  Le LLM
          reçoit donc :

            {ancien résumé cumulatif} + {recent_window nouveaux tours à résumer}

          et produit directement le *nouveau résumé cumulatif*.  Un seul
          appel LLM par cycle, pas deux.

        Déclenchement :
          La compression n'a lieu QUE lorsque le nombre de tours accumulés
          atteint ``recent_window × _COMPRESS_THRESHOLD`` (défaut : 12).
          Avant ce seuil, la méthode ne fait rien — les tours s'accumulent
          sans coût.

        Élagage :
          Après compression, les *recent_window* tours les plus anciens sont
          SUPPRIMÉS de ``self._turns``.  La mémoire est donc bornée à
          ``recent_window × _COMPRESS_THRESHOLD`` tours (12 max pour window=6).

        Si *llm_summarize_fn* est ``None``, la compression se fait par
        troncature simple (sans LLM).
        """
        threshold = self._recent_window * self._COMPRESS_THRESHOLD
        if len(self._turns) < threshold:
            return

        batch_size = self._recent_window

        # --- Étape 1 : extraire le bloc à résumer ---
        batch = self._turns[:batch_size]
        batch_text = "\n".join(
            f"{'Étudiant' if t['role'] == 'student' else 'Tuteur'} : {t['content']}"
            for t in batch
        )

        # --- Étape 2 : un SEUL appel LLM → résumé cumulatif ---
        if llm_summarize_fn is not None:
            try:
                self._summary = llm_summarize_fn(
                    self._build_compress_prompt(batch_text)
                )
            except Exception:
                # Échec LLM → fallback sur troncature simple du bloc courant
                # (on perd le cumul mais on ne bloque pas la conversation)
                self._summary = self._truncate_summary(batch_text)
        else:
            self._summary = self._truncate_summary(batch_text)

        # --- Étape 3 : élaguer les tours résumés ---
        self._turns = self._turns[batch_size:]

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _build_compress_prompt(self, batch_text: str) -> str:
        """Construit le prompt de fusion pour l'appel LLM de compression.

        Si un résumé antérieur existe, on demande une FUSION (ancien + nouveau).
        Sinon, c'est la première compression : simple résumé du bloc.
        """
        if self._summary:
            return (
                "Voici un RÉSUMÉ EXISTANT d'une conversation pédagogique entre "
                "un étudiant et un tuteur, suivi des NOUVEAUX échanges.  Produis "
                "un résumé CUMULATIF en 2-3 phrases maximum qui INTÈGRE les "
                "nouveaux échanges dans l'ancien résumé.  Ne perds aucune "
                "information clé (sujets abordés, points importants, décisions "
                "pédagogiques).\n\n"
                f"RÉSUMÉ EXISTANT :\n{self._summary}\n\n"
                f"NOUVEAUX ÉCHANGES :\n{batch_text}"
            )
        else:
            return (
                "Résume cette conversation pédagogique entre un étudiant et un "
                "tuteur en 2-3 phrases maximum, en gardant les sujets abordés "
                "et les points clés :\n\n"
                f"{batch_text}"
            )

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
        """Tronque l'historique pour respecter *max_tokens*, en sacrifiant
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

def summarize_with_llm(text: str) -> str:
    """Résume un texte via le LLM (utilise le client Ollama générique).

    Signature compatible avec ConversationMemory.compress() : callable(str) -> str.
    """
    from ..core.llm_client import chat
    return chat(
        "Tu es un assistant qui résume des conversations. "
        "Réponds en 2-3 phrases maximum, en français.",
        text,
        temperature=0,
    )