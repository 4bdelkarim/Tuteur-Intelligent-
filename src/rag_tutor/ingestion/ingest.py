#!/usr/bin/env python3
"""
ingest.py — ORCHESTRATEUR de l'ingestion complete : chunking -> embedding -> indexation.

Ne contient AUCUNE logique metier propre -- compose seulement :
  chunking.chunk_corpus()                    (chunks + metadata + embed_text)
  embeddings.BGEEmbeddings                    (embed_text -> vecteurs)
  vector_store.index_children/save_parents    (persistance Chroma + JSON)

C'est le seul fichier a relancer quand le corpus change. evaluate.py et
le pipeline de requete n'appellent jamais ce fichier -- ils consomment la
base deja indexee, via retriever.py.

Entree attendue : le corpus CANONIQUE normalise (``data/normalized/``), pas la
sortie brute d'extraction (``data/processed/``).

Usage :
  python -m rag_tutor.ingestion.ingest data/normalized/
  python -m rag_tutor.ingestion.ingest data/normalized/ --child-target 400 --child-max 750
"""

import argparse
from pathlib import Path

from ..core.chunking import chunk_corpus, print_stats, CHILD_TARGET, CHILD_MAX, CHILD_OVERLAP
from ..core.embeddings import BGEEmbeddings, EMBEDDING_MODEL
from ..core.vector_store import index_children, save_parents


def run(path: Path, child_target: int = CHILD_TARGET, child_max: int = CHILD_MAX,
        child_overlap: int = CHILD_OVERLAP, reset: bool = True):
    """Execute l'ingestion complete : chunking -> embeddings -> indexation Chroma.

    Args:
        path: Dossier (ou fichier) du corpus normalise a indexer.
        child_target: Taille cible (en caracteres) d'un chunk enfant.
        child_max: Taille maximale (hardmax, en caracteres) d'un chunk enfant.
        child_overlap: Chevauchement (en caracteres) ajoute a l'``embed_text``
            des enfants consecutifs (le ``text`` brut reste sans overlap).
        reset: Si vrai, vide la collection Chroma avant indexation.

    Returns:
        Tuple ``(parents, children)`` : les sections parentes remontees au
        runtime et les chunks enfants indexes.
    """
    parents, children = chunk_corpus(path, child_target=child_target, child_max=child_max,
                                      child_overlap=child_overlap)
    print_stats(parents, children)

    embedder = BGEEmbeddings(EMBEDDING_MODEL)
    vectors = embedder.embed_documents([c["embed_text"] for c in children])

    index_children(children, vectors, reset=reset)
    pstore = save_parents(parents)

    print(f"\nOK  {len(children)} enfants indexes  |  {len(parents)} parents -> {pstore}")
    return parents, children


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingestion complete : chunking -> embedding -> indexation Chroma.")
    ap.add_argument("path", help="dossier (ou fichier) de sortie de normalizer.py")
    ap.add_argument("--child-target", type=int, default=CHILD_TARGET)
    ap.add_argument("--child-max", type=int, default=CHILD_MAX)
    ap.add_argument("--child-overlap", type=int, default=CHILD_OVERLAP,
                    help=f"chevauchement entre enfants consecutifs en embed_text (defaut {CHILD_OVERLAP})")
    args = ap.parse_args()

    run(Path(args.path), args.child_target, args.child_max, args.child_overlap)
