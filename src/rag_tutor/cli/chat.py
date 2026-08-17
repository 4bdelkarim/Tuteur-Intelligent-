#!/usr/bin/env python3
"""
chat.py — pose des questions au tuteur RAG directement en ligne de commande,
sans passer par l'evaluation Ragas -- usage interactif normal / demo rapide.

Usage :
  python -m rag_tutor.cli.chat
  python -m rag_tutor.cli.chat --k 6 --show-sources
  python -m rag_tutor.cli.chat --no-query-processing --no-refusal-gate   # mode degrade, plus rapide

  "quit" / "exit" / "q" / Ctrl+C / Ctrl+D pour sortir.
"""

import argparse
import time

from ..core.pipeline import answer as pipeline_answer
from ..core.generator import citation_label


def format_sources(hits: list[dict], max_sources: int = 3) -> str:
    """Affiche les sources des passages recuperes (meta.source/source_url si
    disponible). Defensif : ne plante pas si la forme des hits differe."""
    if not hits:
        return "  (aucune source)"
    lines = []
    for h in hits[:max_sources]:
        meta = h.get("meta", {}) if isinstance(h, dict) else {}
        lines.append(f"  - [{citation_label(meta)}]")
    return "\n".join(lines)


def main() -> None:
    """Point d'entree CLI du Q&A direct (cf. ``rag-chat`` / ``make chat``)."""
    ap = argparse.ArgumentParser(description="Pose des questions au tuteur RAG en interactif.")
    ap.add_argument("--k", type=int, default=4, help="nb de passages recuperes par (sous-)question (defaut 4)")
    ap.add_argument("--no-query-processing", action="store_true",
                    help="saute la reformulation/decomposition de la question (1 appel LLM en moins)")
    ap.add_argument("--no-refusal-gate", action="store_true",
                    help="desactive la decision de refus explicite (comportement implicite au prompt seul)")
    ap.add_argument("--show-sources", action="store_true", help="affiche les sources utilisees apres chaque reponse")
    args = ap.parse_args()

    print("Tuteur RAG -- pose ta question ('quit'/'exit' pour sortir)\n")

    while True:
        try:
            question = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        t0 = time.time()
        result = pipeline_answer(
            question,
            k=args.k,
            use_query_processing=not args.no_query_processing,
            use_refusal_gate=not args.no_refusal_gate,
        )
        elapsed = time.time() - t0

        print(f"\nTuteur ({elapsed:.1f}s) > {result.answer}\n")
        if args.show_sources:
            print("Sources :")
            print(format_sources(result.hits))
            print()


if __name__ == "__main__":
    main()