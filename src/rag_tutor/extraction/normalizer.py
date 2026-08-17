#!/usr/bin/env python3
"""
normalizer.py — Unification des .md (PDF ancien/recent + web) vers UN SEUL format canonique.

Objectif : que le chunker parent-child n'ait plus JAMAIS a detecter quel pipeline a
produit un fichier. Ce script fait ce travail UNE fois, en amont, et ecrit une version
propre et uniforme. Separe du chunking (une responsabilite = un script).

Formats reels absorbes en entree :
  - PDF ancien (pdf_to_md.py)       : <!-- page N -->, --- [INTERPRETATION DE LA FIGURE]
                                       --- ... [FIN INTERPRETATION] ---, titres bruts
                                       numerotes SANS '#', pas de front-matter YAML.
  - PDF recent (pdf_extractor.py) : <!-- loc page=N -->, --- [FIGURE] ---
                                       ... --- [/FIGURE] ---, titres markdown '#'..'###',
                                       front-matter YAML deja present.
  - Web (clean_web_markdown.py)     : front-matter YAML, titres markdown natifs, pas de
                                       page ni de figure (images deja supprimees).

Format canonique en sortie (TOUJOURS) :
  - front-matter YAML complet (source_type, source, + source_url si web)
  - pages   : <!-- loc page=N -->                         (absent si web)
  - figures : --- [FIGURE] --- ... --- [/FIGURE] ---        (absent si web)
  - titres  : '#'..'######' markdown (profondeur deduite du prefixe numerote si present)
  - tables  : markdown pipe (HTML <table> converti si necessaire)
  - nettoyage : lignes '|' orphelines (residu de galerie web non liee a une image),
    normalisation Unicode NFC, espaces.

Usage :
  python -m rag_tutor.extraction.normalizer ./bruts/ --out ./processed/
"""

import re
import argparse
import unicodedata
from pathlib import Path

try:
    from ._html_table import html_table_to_md
except ImportError:  # execute comme script autonome (python normalizer.py)
    from _html_table import html_table_to_md

# =====================================================
# FRONT-MATTER
# =====================================================

FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)
OLD_PDF_COMMENT_RE = re.compile(r'^<!--\s*source:\s*(\S+)\s*\|.*-->\s*\n?', re.MULTILINE)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Separe le front-matter YAML du corps du document.

    Args:
        text: Contenu complet d'un fichier .md (front-matter + corps).

    Returns:
        Tuple ``(meta, corps)`` — ``meta`` est ``{}`` si aucun front-matter
        n'est present (ancien format ``pdf_to_md.py``).
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        import yaml
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}
    return meta, text[m.end():]


def detect_source_type(meta: dict, raw_text: str) -> str:
    """Determine le type de source d'un fichier.

    Priorite au front-matter ; sinon detection heuristique : commentaire
    ``<!-- source: ... -->`` ou marqueurs de page -> ``pdf``, sinon ``web``.

    Args:
        meta: Metadata extraites du front-matter.
        raw_text: Contenu brut complet du fichier.

    Returns:
        ``"pdf"`` ou ``"web"``.
    """
    st = meta.get("source_type")
    if st in ("pdf", "web"):
        return st
    if OLD_PDF_COMMENT_RE.search(raw_text):
        return "pdf"
    return "pdf" if re.search(r'(?im)^<!--\s*(?:loc\s+)?page[=\s]', raw_text) else "web"


def render_frontmatter(meta: dict, source_type: str, source: str) -> str:
    """Reconstruit un front-matter YAML canonique.

    Args:
        meta: Metadata d'origine (copiees, non mutees).
        source_type: ``"pdf"`` ou ``"web"``.
        source: Nom de la source (stem du fichier).

    Returns:
        Bloc front-matter ``--- ... ---`` pret a ecrire en tete de fichier.
    """
    meta = dict(meta)
    meta["source_type"] = source_type
    meta.setdefault("source", source)
    import yaml
    yml = yaml.dump(meta, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yml}\n---\n"


# =====================================================
# PAGES : unification vers <!-- loc page=N -->
# =====================================================

PAGE_OLD_RE = re.compile(r'(?im)^<!--\s*page\s+(\d+)\s*-->\s*$', re.MULTILINE)
PAGE_NEW_RE = re.compile(r'(?im)^<!--\s*loc\s+page=(\d+)\s*-->\s*$', re.MULTILINE)


def unify_pages(text: str) -> str:
    """Convertit '<!-- page N -->' (ancien) -> '<!-- loc page=N -->' (canonique).

    Args:
        text: Corps du document.

    Returns:
        Texte avec la convention de page canonique. Idempotent.
    """
    return PAGE_OLD_RE.sub(lambda m: f"<!-- loc page={m.group(1)} -->", text)


# =====================================================
# FIGURES : unification vers --- [FIGURE] --- ... --- [/FIGURE] ---
# =====================================================

FIG_OLD_RE = re.compile(
    r'---\s*\[INTERPRETATION DE LA FIGURE\]\s*---\s*\n(.*?)\n?---\s*\[FIN INTERPRETATION\]\s*---',
    re.DOTALL | re.IGNORECASE,
)


def unify_figures(text: str) -> str:
    """Convertit le bloc figure ancien -> bloc figure recent.

    Args:
        text: Corps du document.

    Returns:
        Texte avec des blocs ``--- [FIGURE] --- ... --- [/FIGURE] ---``.
        Contenu preserve tel quel, idempotent.
    """
    def repl(m):
        inner = m.group(1).strip()
        return f"--- [FIGURE] ---\n{inner}\n--- [/FIGURE] ---"
    return FIG_OLD_RE.sub(repl, text)


# =====================================================
# TITRES : unification vers markdown '#'..'######'
# =====================================================

# Titre ANCIEN (brut, sans '#') : "1- Introduction", "1.1- Inspiration biologique",
# variante sous-sous-section "3.10.2-.1 ...".
HEADER_OLD_RE = re.compile(r'^(\d+(?:\.\d+)*)-(?:\.(\d+))?\s+(\S.*)$')
# Un vrai titre ne se termine pas par un numero isole (sinon = ligne de sommaire/TOC)
TOC_TAIL_RE = re.compile(r'\d\s*$')
HEADER_MD_RE = re.compile(r'^#{1,6}\s+\S')


def _heading_level(num_prefix: str, has_subindex: bool) -> int:
    depth = num_prefix.count(".") + 1        # "1" ->1 | "1.1" ->2 | "3.10.2" ->3
    if has_subindex:                          # variante "3.10.2-.1"
        depth += 1
    return min(depth + 1, 6)                 # niveau 1 -> ## ; plafond ######


def unify_headers(text: str) -> str:
    """Convertit les titres bruts numerotes en titres markdown.

    Les titres deja au format markdown ne sont PAS touches (idempotent). Les
    lignes de sommaire (terminees par un numero de page isole) sont laissees
    telles quelles -> jamais transformees en faux titres.

    Args:
        text: Corps du document.

    Returns:
        Texte avec des titres ``#``..``######`` dont la profondeur est deduite
        du prefixe numerote.
    """
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if HEADER_MD_RE.match(stripped):
            out.append(line); continue                      # deja markdown -> inchange
        m = HEADER_OLD_RE.match(stripped)
        if m and not TOC_TAIL_RE.search(stripped):
            level = _heading_level(m.group(1), bool(m.group(2)))
            out.append(f"{'#' * level} {stripped.strip()}\n")
        else:
            out.append(line)
    return "".join(out)


# =====================================================
# TABLES : HTML -> markdown (si jamais un ancien PDF en contient)
# =====================================================

HTML_TABLE_RE = re.compile(r'<table\b.*?</table>', re.DOTALL | re.IGNORECASE)


def unify_tables(text: str) -> str:
    """Convertit ``<table>…</table>`` en table Markdown pipe.

    Args:
        text: Corps du document.

    Returns:
        Texte avec des tables Markdown pipe. Idempotent.
    """
    return HTML_TABLE_RE.sub(lambda m: html_table_to_md(m.group(0)), text)


# =====================================================
# NETTOYAGE : residus de galerie web + Unicode/espaces
# =====================================================

ORPHAN_PIPE_RE = re.compile(r'(?m)^[ \t]*\|[ \t]*$\n?')


def strip_orphan_pipes(text: str) -> str:
    """Supprime les lignes reduites a un seul ``|`` isole.

    Args:
        text: Corps du document.

    Returns:
        Texte sans pipes orphelins (residus de galeries web). Les vraies lignes
        de table ``| a | b |`` ne sont pas touchees.
    """
    return ORPHAN_PIPE_RE.sub('', text)


def tidy(text: str) -> str:
    """Normalise Unicode (NFC) et les espaces de fin de ligne.

    Args:
        text: Texte a nettoyer.

    Returns:
        Texte normalise NFC, sans espaces de fin de ligne, blocs de lignes
        vides reduits a deux sauts de ligne maximum.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + "\n"


# =====================================================
# PIPELINE
# =====================================================

def normalize_body(text: str) -> str:
    """Applique toutes les etapes d'unification sur un corps de document.

    Args:
        text: Corps du document (sans front-matter).

    Returns:
        Corps unifie au format canonique (pages, figures, titres, tables,
        nettoyage final).
    """
    text = unify_pages(text)
    text = unify_figures(text)
    text = unify_tables(text)
    text = unify_headers(text)
    text = strip_orphan_pipes(text)
    text = tidy(text)
    return text


def normalize_file(path: Path, out_dir: Path) -> str:
    """Unifie UN fichier .md et ecrit le resultat dans ``out_dir``.

    Args:
        path: Fichier .md source.
        out_dir: Dossier de destination (cree si absent).

    Returns:
        Le ``source_type`` detecte (``"pdf"`` ou ``"web"``).
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(raw)
    source_type = detect_source_type(meta, raw)
    body = normalize_body(body)
    out = render_frontmatter(meta, source_type, path.stem) + "\n" + body
    out_path = out_dir / path.name
    out_path.write_text(out, encoding="utf-8")
    return source_type


def main() -> None:
    ap = argparse.ArgumentParser(description="Unifie les .md (pdf ancien/recent + web) en un seul format.")
    ap.add_argument("input_dir")
    ap.add_argument("--out", default="processed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input_dir).glob("*.md"))
    if not files:
        print(f"Aucun .md dans {args.input_dir}/"); return

    n_ok, n_fail = 0, 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ...", end=" ", flush=True)
        try:
            st = normalize_file(f, out_dir)
        except Exception as e:
            n_fail += 1
            print(f"ECHEC ({type(e).__name__}: {e})")
            continue
        n_ok += 1
        print(f"OK ({st})")

    print(f"\nTermine -> {out_dir}/  ({n_ok} ok, {n_fail} echec(s) sur {len(files)})")


if __name__ == "__main__":
    main()
