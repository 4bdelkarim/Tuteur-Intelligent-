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

Usage :
  python ingest.py ./processed/
  python ingest.py ./processed/ --child-target 400 --child-max 750
"""

import argparse
from pathlib import Path

from ..core.chunking import chunk_corpus, print_stats, CHILD_TARGET, CHILD_MAX, CHILD_OVERLAP
from ..core.embeddings import BGEEmbeddings, EMBEDDING_MODEL
from ..core.vector_store import index_children, save_parents


def run(path, child_target=CHILD_TARGET, child_max=CHILD_MAX, child_overlap=CHILD_OVERLAP,
        reset=True):
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
