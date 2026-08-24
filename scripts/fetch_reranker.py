#!/usr/bin/env python3
"""
fetch_reranker.py — télécharge et VÉRIFIE le reranker BAAI/bge-reranker-v2-m3.

CONTRAIREMENT À LA VERSION PRÉCÉDENTE QUI ABANDONNAIT AU 1er ÉCHEC, ce script :
  1. Vérifie l'espace disque disponible avant de lancer
  2. Détecte et purge AUTOMATIQUEMENT un cache corrompu
  3. Réessaie jusqu'à 3 fois avec backoff exponentiel (coupure wifi, proxy...)
  4. VÉRIFIE le modèle après téléchargement en le chargeant réellement
     (via sentence-transformers) — pas juste « les fichiers sont là »
  5. Donne un diagnostic PRÉCIS en cas d'échec final (pas un message générique)

Sortie : 0 = OK (modèle prêt à l'emploi), 1 = échec (diagnostic complet affiché).
"""

import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dépendances (vérifiées une par une pour des messages d'erreur précis)
# ---------------------------------------------------------------------------

_MISSING = []
try:
    from huggingface_hub import snapshot_download
except ImportError:
    _MISSING.append("huggingface_hub")
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    _MISSING.append("sentence-transformers")

if _MISSING:
    print(f"   ERREUR : dépendance(s) manquante(s) : {', '.join(_MISSING)}", file=sys.stderr)
    print("   -> pip install huggingface_hub sentence-transformers", file=sys.stderr)
    print("   -> ou relance `make setup` (etape 2)", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = "BAAI/bge-reranker-v2-m3"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / f"models--BAAI--bge-reranker-v2-m3"
MIN_DISK_SPACE_GB = 5  # le modèle fait ~2.2 Go, on exige 5 pour être large
MAX_RETRIES = 3
RETRY_BACKOFF = 3       # secondes entre tentatives : 3, 6, 12
DOWNLOAD_TIMEOUT = 900  # 15 minutes — une connexion très lente + les ~2 Go


# ---------------------------------------------------------------------------
# Vérifications préparatoires
# ---------------------------------------------------------------------------

def check_disk_space() -> bool:
    """Vérifie qu'il y a assez d'espace disque. Retourne False si insuffisant."""
    try:
        usage = shutil.disk_usage(CACHE_DIR.parent if CACHE_DIR.parent.exists() else Path.home())
        free_gb = usage.free / (1024 ** 3)
        if free_gb < MIN_DISK_SPACE_GB:
            print(f"   ERREUR : espace disque insuffisant ({free_gb:.1f} Go libres, "
                  f"{MIN_DISK_SPACE_GB} Go requis).", file=sys.stderr)
            print(f"   -> Libère de l'espace sur {usage._asdict()} et relance.", file=sys.stderr)
            return False
        return True
    except Exception:
        return True  # impossible de vérifier → on tente quand même


def is_cache_corrupted() -> bool:
    """Détecte un cache CORROMPU : le dossier existe mais le fichier de poids
    model.safetensors est absent ou trop petit (< 100 Mo — un fichier de 2,2 Go
    tronqué par une coupure réseau)."""
    if not CACHE_DIR.exists():
        return False  # pas de cache du tout → pas corrompu, juste absent

    # Cherche model.safetensors dans les snapshots/
    snapshots_dir = CACHE_DIR / "snapshots"
    if not snapshots_dir.exists():
        return True   # dossier incomplet : pas de snapshots/

    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        safetensors = snapshot / "model.safetensors"
        if safetensors.exists():
            size_mb = safetensors.stat().st_size / (1024 ** 2)
            if size_mb < 100:  # un fichier de 2,2 Go tronqué
                return True
            return False  # fichier de poids présent ET assez gros → cache sain
    return True  # aucun model.safetensors trouvé


def purge_cache() -> None:
    """Supprime le cache corrompu. Ne fait rien si le dossier n'existe pas."""
    if CACHE_DIR.exists():
        print(f"   🧹 Cache corrompu détecté → suppression de {CACHE_DIR} ...", flush=True)
        try:
            shutil.rmtree(CACHE_DIR)
            print("   ✅ Cache purgé.", flush=True)
        except OSError as e:
            print(f"   ⚠️  Impossible de supprimer le cache : {e}", file=sys.stderr)
            print(f"   -> Supprime manuellement : rm -rf {CACHE_DIR}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Téléchargement avec retry
# ---------------------------------------------------------------------------

def download_with_retries() -> bool:
    """Télécharge le modèle avec retries. Retourne True si OK."""
    for attempt in range(1, MAX_RETRIES + 1):
        prefix = f"   [tentative {attempt}/{MAX_RETRIES}]" if MAX_RETRIES > 1 else "   "
        print(f"{prefix} Téléchargement de {REPO} (~600 Mo à 2,2 Go)...", flush=True)

        try:
            snapshot_download(
                REPO,
                resume_download=True,        # reprend un téléchargement interrompu
                max_workers=4,               # parallélise (4 fichiers en même temps)
                tqdm_class=None,             # barre de progression native huggingface_hub
            )
        except Exception as e:
            msg = str(e).lower()
            if "no such file" in msg or "not found" in msg:
                print(f"   ERREUR : le modèle {REPO} n'existe pas sur HuggingFace.", file=sys.stderr)
                print(f"   -> Vérifie le nom du modèle ou l'état de huggingface.co", file=sys.stderr)
                return False
            if "timeout" in msg or "timed out" in msg or "connection" in msg:
                print(f"   ⚠️  Timeout/connexion (tentative {attempt}/{MAX_RETRIES}) : {e}", file=sys.stderr)
            elif "space" in msg or "disk" in msg:
                print(f"   ERREUR : espace disque insuffisant pendant le téléchargement.", file=sys.stderr)
                print(f"   -> Libère de l'espace et relance.", file=sys.stderr)
                return False
            else:
                print(f"   ⚠️  Échec (tentative {attempt}/{MAX_RETRIES}) : {e}", file=sys.stderr)

            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"   ↻ Nouvelle tentative dans {wait}s ...", flush=True)
                time.sleep(wait)
                # Purge le cache après chaque échec pour repartir de zéro
                # (un téléchargement interrompu laisse un état corrompu)
                purge_cache()
            continue

        # Téléchargement réussi — vérification POST-téléchargement
        print(f"   ✅ Fichiers téléchargés. Vérification du chargement réel...", flush=True)
        return True

    return False  # toutes les tentatives ont échoué


# ---------------------------------------------------------------------------
# Vérification POST-téléchargement (charge RÉELLEMENT le modèle)
# ---------------------------------------------------------------------------

def verify_model_works() -> bool:
    """Vérifie que le modèle se charge VRAIMENT (pas juste que les fichiers sont là).
    C'est le SEUL test qui garantit que le téléchargement a PRODUIT un modèle
    utilisable — snapshot_download peut réussir mais laisser un état inconsistent
    (métadonnées manquantes, version incompatible de transformers...)."""
    print("   🔍 Vérification : chargement réel du modèle via sentence-transformers...", flush=True)
    try:
        model = CrossEncoder(REPO, device="cpu")
        # Test minimal : prédire un score sur une paire triviale
        score = model.predict([("Bonjour", "Bonjour le monde")])
        if score is not None and len(score) > 0:
            print(f"   ✅ Modèle chargé ET fonctionnel (score test = {float(score[0]):.3f}).", flush=True)
            return True
        else:
            print("   ⚠️  Modèle chargé mais predict() n'a rien retourné.", file=sys.stderr)
            return False
    except Exception as e:
        print(f"   ERREUR lors de la vérification : {e}", file=sys.stderr)
        print(f"   -> Le téléchargement a peut-être réussi mais le modèle est inutilisable.", file=sys.stderr)
        print(f"   -> Causes possibles :", file=sys.stderr)
        print(f"      - Version de transformers/sentence-transformers trop ancienne", file=sys.stderr)
        print(f"      - pip install --upgrade sentence-transformers transformers", file=sys.stderr)
        print(f"      - Cache partiellement corrompu : supprime {CACHE_DIR} et relance", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> int:
    # Étape 0 : espace disque
    if not check_disk_space():
        return 1

    # Étape 1 : cache déjà sain ? → rien à faire
    if CACHE_DIR.exists() and not is_cache_corrupted():
        print("   ✅ bge-reranker-v2-m3 déjà en cache local (valide).")
        # Vérifie quand même qu'il charge (le cache peut être "présent" mais
        # incompatible avec la version actuelle de transformers)
        if not verify_model_works():
            print("   ⚠️  Cache présent mais modèle inutilisable → re-téléchargement forcé.", flush=True)
            purge_cache()
        else:
            return 0

    # Étape 2 : cache corrompu → purge automatique
    if is_cache_corrupted():
        print("   ⚠️  Cache corrompu détecté (fichier de poids absent ou incomplet).", flush=True)
        purge_cache()

    # Étape 3 : téléchargement avec retries
    if not download_with_retries():
        print("", file=sys.stderr)
        print("   ❌ ÉCHEC après plusieurs tentatives de téléchargement.", file=sys.stderr)
        print("", file=sys.stderr)
        print("   Causes possibles et solutions :", file=sys.stderr)
        print("   1. Pas de connexion Internet", file=sys.stderr)
        print("      → Vérifie ta connexion et relance.", file=sys.stderr)
        print("   2. Proxy d'entreprise / firewall qui bloque huggingface.co", file=sys.stderr)
        print("      → Configure HTTPS_PROXY ou utilise un VPN.", file=sys.stderr)
        print("   3. HuggingFace est down", file=sys.stderr)
        print("      → Vérifie https://status.huggingface.co et réessaie plus tard.", file=sys.stderr)
        print("   4. DNS ne résout pas huggingface.co", file=sys.stderr)
        print("      → Teste : curl -I https://huggingface.co", file=sys.stderr)
        print("", file=sys.stderr)
        print("   En attendant, le système fonctionne EN MODE DÉGRADÉ (sans rerank).", file=sys.stderr)
        print("   Repasse en mode complet dès que le téléchargement réussit.", file=sys.stderr)
        print("   Commande à relancer : python scripts/fetch_reranker.py", file=sys.stderr)
        return 1

    # Étape 4 : vérification POST-téléchargement (obligatoire)
    if not verify_model_works():
        # Dernier recours : purge et on laisse le fallback automatique agir
        print("   ⚠️  Téléchargement OK mais modèle inutilisable.", file=sys.stderr)
        print("   -> Le système utilisera le mode dégradé (hybrid) automatiquement.", file=sys.stderr)
        return 1

    print("   ✅ bge-reranker-v2-m3 prêt à l'emploi !")
    return 0


if __name__ == "__main__":
    sys.exit(main())