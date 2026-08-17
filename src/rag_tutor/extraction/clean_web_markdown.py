#!/usr/bin/env python3
"""
clean_web_markdown.py — nettoyage web POST-extraction, AVANT chunking.

Corrige 5 problemes reels du script d'origine (voir chaque fonction ci-dessous
pour le detail) :
  1) front-matter garde INLINE dans le .md (pas de sidecar .yaml) -> source_type
     et les autres champs voyagent AVEC le contenu jusqu'au chunker, comme pour
     le pipeline PDF.
  2) TOUTES les images sont supprimees (valides ou non) ainsi que tous les
     liens externes -> seul le texte/legende autour est conserve. Les galeries
     en table (valides ou cassees) sont d'abord converties en une ligne de
     legende lisible, AVANT suppression, pour ne jamais laisser de pipes
     orphelines (bug de l'original qui ne traitait que les galeries VIDES).
  4) normalisation des maths : \\[...\\] -> $$...$$, \\(...\\) -> $...$ (delimiteurs
     NYU unifies avec la convention $$ deja utilisee cote PDF), + dictionnaire
     de macros custom courantes sur ces cours (\\vx, \\mX, \\mY... -> \\mathbf{}).
  5) suppression des exercices BORNEE (s'arrete a la prochaine section de meme
     niveau ou a la fin du texte, jamais un DOTALL non borne) ; ACTIVEE par
     defaut (les exercices sans solution n'ont pas de valeur de recuperation
     pour le RAG) -> --keep-exercises pour les conserver si besoin sur un
     corpus precis.

Usage :
  python clean_web_markdown.py markdown_raw/ --out markdown_clean/
  python clean_web_markdown.py markdown_raw/ --out markdown_clean/ --keep-exercises
"""
import re
import argparse
import unicodedata
from pathlib import Path

import yaml

# =====================================================
# 1) FRONT-MATTER — reste DANS le .md (pas de sidecar)
# =====================================================

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Separe le front-matter YAML du corps du document.

    Le front-matter reste INLINE dans le .md (pas de fichier sidecar) : le
    dictionnaire est reinjecte en sortie par :func:`render_frontmatter`.

    Args:
        text: Contenu complet d'un fichier .md (front-matter + corps).

    Returns:
        Tuple ``(meta, corps)`` — ``meta`` est un dict vide ``{}`` si aucun
        front-matter n'est present.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, text[m.end():]


def render_frontmatter(meta: dict) -> str:
    meta = dict(meta)
    meta.setdefault("source_type", "web")           # schema unifie avec le pdf
    yml = yaml.dump(meta, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yml}\n---\n"


# =====================================================
# 2+3) IMAGES — SUPPRESSION TOTALE (valides ou non), sans artefact residuel
# =====================================================

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
# bloc : N lignes d'images (VALIDES OU VIDES) consecutives, suivies d'une ligne
# de legende "| ... | ... |" -> matche desormais TOUTE galerie (plus seulement
# les cassees), puisqu'on supprime les images dans tous les cas maintenant.
_GALLERY_RE = re.compile(
    r"(?:^!\[[^\]]*\]\([^)]*\)\s*\|?\s*\n)+^\|(.+)\|\s*$",
    re.MULTILINE,
)


def convert_image_galleries_to_captions(text: str) -> str:
    """Un groupe d'images en table (galerie), valides ou non, + sa ligne de
    legende -> UNE ligne de legende lisible, images supprimees. Doit tourner
    AVANT toute suppression generique d'image -> sinon la table se retrouve
    avec des cellules vides et des pipes orphelines (bug de l'original)."""
    def repl(m):
        cells = [c.strip() for c in m.group(1).split("|") if c.strip()]
        return "*(figure — " + " ; ".join(cells) + ")*\n"
    return _GALLERY_RE.sub(repl, text)


def remove_all_images(text: str) -> str:
    """Supprime TOUTES les images restantes (valides ou non), isolees (hors
    galerie deja convertie ci-dessus). Le texte/legende autour est conserve
    tel quel (ex. 'Figure 2: Network Architecture' reste, sans l'image)."""
    return _IMG_RE.sub("", text)


# =====================================================
# LIENS / EMOJIS — inchanges (corrects dans l'original)
# =====================================================

def strip_links_keep_text(text: str) -> str:
    """[texte](url) -> texte. Le lookbehind (?<!!) exclut les IMAGES ![alt](src),
    qui commencent par '!' -> sinon leurs crochets/parentheses sont mutiles."""
    return re.sub(r"(?<!!)\[([^\]]+)\]\([^\)]*\)", r"\1", text)


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F"           # variation selector-16 (accole aux emojis, ex. 🎙️)
    "\U0000200D"           # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(text: str) -> str:
    """Supprime les emojis et autres caracteres Unicode decoratifs.

    Args:
        text: Markdown a nettoyer.

    Returns:
        Texte sans emojis (les doubles espaces residuels sont reduits a un).
    """
    return _EMOJI_RE.sub("", text).replace("  ", " ")


# =====================================================
# 4) MATHS — unification des delimiteurs + macros (ETAPE MANQUANTE de l'original)
# =====================================================

# Macros customs frequentes sur ces cours (Alfredo Canziani / NYU) : vecteurs
# prefixe 'v', matrices prefixe 'm'. Liste extensible sans risque (si absente,
# le texte reste inchange).
LATEX_MACROS = {
    r"\vx": r"\mathbf{x}", r"\vy": r"\mathbf{y}", r"\vz": r"\mathbf{z}",
    r"\mX": r"\mathbf{X}", r"\mY": r"\mathbf{Y}", r"\mW": r"\mathbf{W}",
}


def unify_math(text: str) -> str:
    """\\[...\\] -> $$...$$ ; \\(...\\) -> $...$ ; expanse les macros connues.
    Idempotent sur du LaTeX deja en $/$$-delimiteurs (ex. d2l) -> ne les touche pas."""
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    for macro, expansion in LATEX_MACROS.items():
        text = text.replace(macro, expansion)
    return text


# =====================================================
# 5) EXERCICES — BORNEE, DESACTIVEE PAR DEFAUT (choix pedagogique explicite)
# =====================================================

# S'arrete a la prochaine ligne de titre (#+) ou a la fin du texte -> jamais
# tout le reste du document (bug de l'original avec re.DOTALL non borne).
# [\d.]* autorise un prefixe numerote ("14.10.5. Exercises", pas juste "Exercises").
_EXERCISE_SECTION_RE = re.compile(
    r"^#+\s*[\d.]*\s*(Exercises?|Problems?)\s*\n.*?(?=^#+\s|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def remove_exercises(text: str) -> str:
    """Supprime les sections Exercises/Problems, bornees a la section suivante.

    Args:
        text: Markdown a nettoyer.

    Returns:
        Texte sans les sections d'exercices.
    """
    return _EXERCISE_SECTION_RE.sub("", text)


# =====================================================
# NETTOYAGE FINAL — Unicode NFC + espaces (dernier, comme _tidy cote pdf)
# =====================================================

def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# =====================================================
# PIPELINE
# =====================================================

def clean_markdown_body(text: str, strip_exercises: bool = True) -> str:
    text = convert_image_galleries_to_captions(text)   # AVANT toute suppression generique
    text = remove_all_images(text)
    text = unify_math(text)
    text = strip_links_keep_text(text)
    text = remove_emojis(text)
    if strip_exercises:
        text = remove_exercises(text)
    text = normalize_whitespace(text)
    return text


def process_file(path: Path, out_dir: Path, strip_exercises: bool = True) -> None:
    """Nettoie UN fichier .md web et ecrit le resultat dans ``out_dir``.

    Args:
        path: Fichier .md source.
        out_dir: Dossier de destination (cree si absent).
        strip_exercises: Si True, supprime les sections Exercises/Problems.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(raw)
    cleaned = clean_markdown_body(body, strip_exercises=strip_exercises)
    out = render_frontmatter(meta) + "\n" + cleaned
    (out_dir / path.name).write_text(out, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Nettoyage markdown web, pret au chunking.")
    ap.add_argument("input_dir")
    ap.add_argument("--out", default="markdown_clean")
    ap.add_argument("--keep-exercises", action="store_true",
                    help="Conserve les sections Exercises/Problems (supprimees par defaut)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input_dir).glob("*.md"))
    if not files:
        print(f"Aucun .md trouve dans {args.input_dir}/"); return

    n_ok, n_fail = 0, 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ...", end=" ", flush=True)
        try:
            process_file(f, out_dir, strip_exercises=not args.keep_exercises)
        except Exception as e:                    # un fichier casse N'ARRETE PLUS le lot
            n_fail += 1
            print(f"ECHEC ({type(e).__name__}: {e})")
            continue
        n_ok += 1
        print("OK")

    print(f"\nTermine -> {out_dir}/  ({n_ok} ok, {n_fail} echec(s) sur {len(files)} fichier(s))")
    if n_fail:
        print("-> relance avec le nom du fichier en echec pour en voir le detail complet :")
        print("   python -c \"import clean_web_markdown as C; C.process_file(__import__('pathlib').Path('FICHIER.md'), __import__('pathlib').Path('.'))\"")


if __name__ == "__main__":
    main()
