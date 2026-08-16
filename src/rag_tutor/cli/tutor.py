#!/usr/bin/env python3
"""
tutor.py — CLI conversationnel pour le tuteur pédagogique socratique v2.

Contrairement à chat.py (Q&A direct, une question → une réponse), ce module
gère une CONVERSATION multi-tours avec :
  - Mémoire conversationnelle (ConversationMemory)
  - Prompt socratique (TUTOR_SYSTEM_PROMPT)
  - Compression automatique de l'historique
  - Évaluation par question (--show-eval)

Usage :
  python -m rag_tutor.cli.tutor
  python -m rag_tutor.cli.tutor --k 6 --show-sources
  python -m rag_tutor.cli.tutor --show-eval           # évaluation complète par question
  python -m rag_tutor.cli.tutor --show-eval --no-judge  # sans le juge LLM (plus rapide)

  "quit" / "exit" / "q" / Ctrl+C / Ctrl+D pour sortir.
  "clear" pour réinitialiser la conversation.
"""

import argparse
import time

from ..core.pipeline import answer as pipeline_answer
from ..core.generator import REFUSAL_MESSAGE, citation_label
from ..conversation.memory import ConversationMemory, summarize_with_llm
from ..conversation.prompts import TUTOR_SYSTEM_PROMPT
from ..evaluation.per_question import evaluate_response, format_eval_report


def format_sources(hits, max_sources=3):
    """Affiche les sources des passages récupérés."""
    if not hits:
        return "  (aucune source)"
    lines = []
    for h in hits[:max_sources]:
        meta = h.get("meta", {}) if isinstance(h, dict) else {}
        lines.append(f"  - [{citation_label(meta)}]")
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
    ap.add_argument("--show-eval", action="store_true",
                    help="affiche l'évaluation détaillée après chaque réponse "
                         "(stats pipeline + juge LLM + retrieval)")
    ap.add_argument("--no-judge", action="store_true",
                    help="désactive le juge LLM dans --show-eval (garde stats pipeline + retrieval)")
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

        # --- Ajouter la question à l'historique (avant génération) ---
        memory.add_turn("student", question)

        # --- Formater l'historique pour le prompt ---
        history_text = memory.get_formatted_history()

        full_answer = ""  # initialisé ici pour éviter UnboundLocalError

        t0 = time.time()
        result = pipeline_answer(
            question,
            k=args.k,
            system_prompt=TUTOR_SYSTEM_PROMPT,
            use_query_processing=not args.no_query_processing,
            use_refusal_gate=not args.no_refusal_gate,
            history=history_text,
            stream=True,
        )
        elapsed = time.time() - t0

        # --- Affichage ---
        if result.refused:
            print(f"\n🤖 Tuteur 🚫 REFUSÉ > {REFUSAL_MESSAGE}")
            memory.add_turn("tutor", REFUSAL_MESSAGE)
        else:
            # Streaming token par token
            print(f"\n🤖 Tuteur > ", end="", flush=True)
            t_start = time.time()
            for token in result.answer_stream:
                print(token, end="", flush=True)
                full_answer += token
            t_end = time.time()
            elapsed_gen = t_end - t_start
            print(f"  ({elapsed_gen:.1f}s)")
            memory.add_turn("tutor", full_answer)

        # --- Rapport d'évaluation par question ---
        if args.show_eval:
            try:
                eval_report = evaluate_response(
                    question, result,
                    answer_text=full_answer if not result.refused else result.answer,
                    run_judge=not args.no_judge,
                    run_retrieval_scores=True,
                )
                print(f"\n{format_eval_report(eval_report)}")
            except Exception as e:
                print(f"\n⚠️  [eval] Erreur lors de l'évaluation : {e}")

        # --- Compression si nécessaire ---
        memory.compress(llm_summarize_fn=summarize_with_llm)

        if args.show_sources:
            print("\n📚 Sources :")
            print(format_sources(result.hits))
            print()

        # --- Indicateur de mémoire ---
        print(f"   [mémoire: {memory.turn_count} tours"
              f"{' | résumé: ' + str(len(memory.summary)) + ' car.' if memory.summary else ''}]")


if __name__ == "__main__":
    main()
