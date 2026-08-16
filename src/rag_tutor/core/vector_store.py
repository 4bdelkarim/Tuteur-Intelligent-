#!/usr/bin/env python3
"""
vector_store.py — SEULE RESPONSABILITE : persister et relire les enfants
(vecteurs + metadata) dans Chroma, et les parents (sections completes) en JSON.

Ni chunking, ni embedding ici -- ce module recoit des enfants deja chunkes
(chunk_parent_child.py) et deja vectorises (embeddings.py), et se contente de
les stocker/relire. Utilise a la fois par ingest.py (ecriture) et par
retriever_hybride.py (lecture) : c'est le SEUL endroit qui connait DB_DIR,
COLLECTION_NAME et le format du fichier parents_*.json -- personne d'autre
ne doit ouvrir Chroma ou ce JSON directement, sinon le format peut diverger
silencieusement entre l'ecriture et la lecture.
"""

import json
from pathlib import Path

DB_DIR          = "chroma_db"
COLLECTION_NAME = "cours_ml_fig"

# ChromaDB refuse un .add() au-dela d'un certain nombre d'elements en un seul appel
# (limite interne variable selon la version, ex. 5461 constate) -> on insere par lots,
# largement en-dessous de cette limite pour rester robuste aux variations de version.
CHROMA_ADD_BATCH_SIZE = 2000


def _parents_path(db_dir=DB_DIR, collection_name=COLLECTION_NAME):
    return Path(db_dir) / f"parents_{collection_name}.json"


def index_children(children, vectors, db_dir=DB_DIR, collection_name=COLLECTION_NAME,
                    batch_size=CHROMA_ADD_BATCH_SIZE, reset=True):
    """Cree (ou reset) la collection Chroma et insere les enfants + leurs vecteurs, par lots.
    `reset=True` (defaut) : supprime et recree la collection -> reindexation complete.
    `reset=False` : ajoute a une collection existante (indexation incrementale)."""
    import chromadb

    client = chromadb.PersistentClient(path=db_dir)
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        coll = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    else:
        coll = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    ids = [c["id"] for c in children]
    documents = [c["text"] for c in children]
    metadatas = [{
        "parent_id": c["parent_id"], "source": c["source"],
        "source_type": c["source_type"],
        "page": c["page"] if c["page"] is not None else -1,
        "section": c["section"], "child_type": c["child_type"],
    } for c in children]

    n = len(ids)
    n_batches = (n + batch_size - 1) // batch_size
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        print(f"  insertion Chroma : lot {i // batch_size + 1}/{n_batches} "
              f"({j - i} elements)...", flush=True)
        coll.add(
            ids=ids[i:j],
            documents=documents[i:j],
            embeddings=vectors[i:j],
            metadatas=metadatas[i:j],
        )
    return coll


def save_parents(parents, db_dir=DB_DIR, collection_name=COLLECTION_NAME):
    """Ecrit le magasin des parents (sections completes) en JSON a cote de la base Chroma."""
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    pstore = _parents_path(db_dir, collection_name)
    pstore.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")
    return pstore


def load_parents(db_dir=DB_DIR, collection_name=COLLECTION_NAME):
    """Relit le magasin des parents -- utilise par le retrieval pour retrouver le texte
    complet d'une section a partir d'un parent_id."""
    return json.loads(_parents_path(db_dir, collection_name).read_text(encoding="utf-8"))


def get_collection(db_dir=DB_DIR, collection_name=COLLECTION_NAME):
    """Ouvre la collection Chroma existante en lecture -- utilise par le retrieval."""
    import chromadb
    return chromadb.PersistentClient(path=db_dir).get_collection(collection_name)