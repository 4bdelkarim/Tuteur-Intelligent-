#!/usr/bin/env python3
"""
tutor.py — CLI conversationnel pour le tuteur pédagogique socratique v2.

Contrairement à chat.py (Q&A direct, une question → une réponse), ce module
gère une CONVERSATION multi-tours avec :
  - Mémoire conversationnelle (ConversationMemory)
  - Détection de changement de sujet → reset du contexte retrieval
  - Prompt socratique (TUTOR_SYSTEM_PROMPT)
  - Compression automatique de l'historique

Usage :
  python -m rag_tutor.cli.tutor
  python -m rag_tutor.cli.tutor --k 6 --show-sources

  "quit" / "exit" / "q" / Ctrl+C / Ctrl+D pour sortir.
  "clear" pour réinitialiser la conversation.
"""

import argparse
import time
import sys

from ..core.pipeline import answer as pipeline_answer
from ..conversation.memory import ConversationMemory
from ..conversation.prompts import TUTOR_SYSTEM_PROMPT


def format_sources(hits, max_sources=3):
    """Affiche les sources des passages récupérés."""
    if not hits:
        return "  (aucune source)"
    lines = []
    for h in hits[:max_sources]:
        meta = h.get("meta", {}) if isinstance(h, dict) else {}
        source = meta.get("source") or meta.get("source_url") or "source inconnue"
        section = meta.get("section", "")
        pages = ""
        if meta.get("page_start"):
            pages = f" (p{meta['page_start']}-{meta.get('page_end', '?')})"
        lines.append(f"  - {source} | {section}{pages}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Tuteur pédagogique socratique — conversation multi-tours."
    )
    ap.add_argument("--k", type=int, default=4,
                    help="nb de passages récupérés par question (défaut 4)")
    ap.add_argument("--no-query-processing", action="store_true",
                    help="saute la reformulation/décomposition (1 appel LLM en moins)")
    ap.add_argument("--no-refusal-gate", action="store_true",
                    help="désactive le refus explicite M1+M2")
    ap.add_argument("--show-sources", action="store_true",
                    help="affiche les sources utilisées après chaque réponse")
    ap.add_argument("--recent-window", type=int, default=6,
                    help="nombre de tours récents gardés intacts (défaut 6)")
    args = ap.parse_args()

    memory = ConversationMemory(recent_window=args.recent_window)

    print("═" * 60)
    print("  TUTEUR PÉDAGOGIQUE — Mode Socratique")
    print("  Tape 'quit' pour sortir, 'clear' pour réinitialiser")
    print("═" * 60)

    while True:
        try:
            question = input("\n🧑 Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Au revoir !")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("👋 Au revoir !")
            break

        if question.lower() == "clear":
            memory.clear()
            print("🧹 Conversation réinitialisée.")
            continue

        # --- Détection changement de sujet ---
        if memory.is_new_topic(question):
            print("🔄 Nouveau sujet détecté — le contexte est rafraîchi.")
            # On garde l'historique mais on pourrait reset le retrieval
            # (pour l'instant, le retrieval repart de zéro à chaque question
            #  donc aucun reset explicite n'est nécessaire)

        # --- Ajouter la question à l'historique (avant génération) ---
        memory.add_turn("student", question)

        # --- Formater l'historique pour le prompt ---
        history_text = memory.get_formatted_history()

        t0 = time.time()
        result = pipeline_answer(
            question,
            k=args.k,
            system_prompt=TUTOR_SYSTEM_PROMPT,
            use_query_processing=not args.no_query_processing,
            use_refusal_gate=not args.no_refusal_gate,
            history=history_text,
        )
        elapsed = time.time() - t0

        # --- Ajouter la réponse à l'historique ---
        memory.add_turn("tutor", result.answer)

        # --- Compression si nécessaire ---
        memory.compress()

        # --- Affichage ---
        status = "🚫 REFUSÉ" if result.refused else f"({elapsed:.1f}s)"
        print(f"\n🤖 Tuteur {status} > {result.answer}")

        if args.show_sources:
            print("\n📚 Sources :")
            print(format_sources(result.hits))
            print()

        # --- Indicateur de mémoire ---
        print(f"   [mémoire: {memory.turn_count} tours"
              f"{' | résumé: ' + str(len(memory.summary)) + ' car.' if memory.summary else ''}]")


if __name__ == "__main__":
    main()
