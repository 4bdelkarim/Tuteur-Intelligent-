#!/usr/bin/env python3
"""
fetch_reranker.py — pré-télécharge le reranker BAAI/bge-reranker-v2-m3 dans le
cache HuggingFace local.

Utilisé par `make setup` (étape _check-reranker). Sans cette étape, le premier
`make chat` déclencherait un téléchargement de ~600 Mo à 1,2 Go SANS barre de
progression (le Makefile exporte TQDM_DISABLE=1) → le chat semble gelé pendant
plusieurs minutes. En pré-téléchargeant ici (TQDM_DISABLE désactivé pour cette
étape), le premier chat est immédiat, et le cache suffit ensuite, même hors-ligne.

Comportement :
  - modèle déjà en cache local  -> ne fait rien (rapide, aucune requête réseau)
  - sinon                       -> télécharge (barre de progression tqdm), sort 0
  - échec réseau                -> message clair + sortie 1 (sans ce modèle, le
                                   mode hybrid_rerank échouerait de toute façon ;
                                   alternative documentée : MODE='hybrid' dans
                                   src/rag_tutor/core/retriever.py)

Sortie : 0 = OK, 1 = échec.
"""

import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - dependances non installees
    print("   ERREUR : huggingface_hub introuvable — l'installation des", file=sys.stderr)
    print("   dependances (etape 2 de make setup) a probablement echoue.", file=sys.stderr)
    sys.exit(1)

REPO = "BAAI/bge-reranker-v2-m3"


def is_cached(repo: str = REPO) -> bool:
    """True si le modele est deja present dans le cache HF (aucune requete reseau)."""
    try:
        return snapshot_download(repo, local_files_only=True) is not None
    except Exception:
        return False


def main() -> int:
    if is_cached():
        print("   OK  bge-reranker-v2-m3 deja en cache local")
        return 0

    print("   Telechargement de bge-reranker-v2-m3 "
          "(~600 Mo a 1,2 Go, une seule fois)...", flush=True)
    try:
        snapshot_download(REPO)
    except Exception as e:
        print(f"   ERREUR : {e}", file=sys.stderr)
        print("   -> une connexion a Hugging Face est requise pour ce modele.", file=sys.stderr)
        print("   -> sans lui, passe en mode degrade : MODE = 'hybrid' dans", file=sys.stderr)
        print("      src/rag_tutor/core/retriever.py", file=sys.stderr)
        return 1

    print("   OK  bge-reranker-v2-m3 telecharge et mis en cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
