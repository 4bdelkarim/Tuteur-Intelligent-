"""
transcribe_figures_v3.py

Détecte les figures composites d'un .md GLM-OCR via le pattern de légende
"FIGURE X-Y - ...", crop chaque région complète depuis le PDF source, envoie
un crop par figure (contenant toutes ses sous-figures) au VLM, splice les
descriptions dans le .md.

Dépendances :
    pip install ollama pymupdf

Usage :
    python scripts/describe.py data/mdfiles/pdf/02_NN.md data/raw/pdf/02_NN.pdf --dry-run
    python transcribe_figures_v3.py cours.md cours.pdf --dry-run    # lister les groupes
    python transcribe_figures_v3.py cours.md cours.pdf --save-crops out/crops  # crops seuls (pas de VLM)
    python transcribe_figures_v3.py cours.md cours.pdf              # pipeline complet
"""

import argparse
import re
import shutil
import sys
import time
from collections import namedtuple
from pathlib import Path

FIG_REF_PATTERN = re.compile(
    r'!\[Image (?P<label>(?P<page>\d+)-(?P<idx>\d+))\]'
    r'\(imgs/cropped_page\d+_idx\d+\.jpg\)'
)

CAPTION_PATTERN = re.compile(
    r'^FIGURE\s+(?P<label>\d+-\d+)\s*[-–]\s*(?P<caption>.+?)$',
    re.MULTILINE,
)

FigureGroup = namedtuple("FigureGroup", "label caption sub_refs page")


# ---------------------------------------------------------------------------
# Parsing du .md
# ---------------------------------------------------------------------------

def parse_figure_groups(md_content: str) -> list:
    """Regroupe les references ![Image ...] en composites via la ligne 'FIGURE X-Y - ...'."""
    groups = []
    pending_refs = []

    for line in md_content.splitlines():
        img_match = FIG_REF_PATTERN.search(line)
        cap_match = CAPTION_PATTERN.match(line.strip())

        if img_match:
            pending_refs.append(img_match)
        elif cap_match and pending_refs:
            pages = {int(m.group("page")) for m in pending_refs}
            # Cas rare : sous-figures étalées sur plusieurs pages -> première page
            page = min(pages)
            groups.append(FigureGroup(
                label=cap_match.group("label"),
                caption=cap_match.group("caption").strip(),
                sub_refs=pending_refs,
                page=page,
            ))
            pending_refs = []

    # Attention : si des refs restent en attente sans légende, on les laisse
    # intactes dans le .md (edge case, à traiter manuellement)
    return groups


# ---------------------------------------------------------------------------
# Cropping via PyMuPDF
# ---------------------------------------------------------------------------

def find_figure_bbox(page, caption_text: str):
    """Bbox du crop = espace au-dessus de la légende jusqu'au bloc texte précédent."""
    import fitz

    caption_rects = page.search_for(caption_text)
    if not caption_rects:
        return None
    caption_bbox = caption_rects[0]

    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    above = [
        b for b in blocks
        if b[3] < caption_bbox.y0 - 5
        and b[4].strip()
        and caption_text.strip() not in b[4].strip()
    ]

    if above:
        closest = max(above, key=lambda b: b[3])
        top = closest[3] + 5
    else:
        top = page.rect.y0 + 20

    left = page.rect.x0 + 10
    right = page.rect.x1 - 10
    bottom = caption_bbox.y0 - 5

    if bottom - top < 30:
        return None  # bbox dégénérée

    return fitz.Rect(left, top, right, bottom)


def crop_figure(pdf_path: Path, page_num: int, figure_label: str,
                out_path: Path, dpi: int = 200) -> bool:
    """Crop la figure composite. True si crop précis, False si fallback page complète."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        if not 1 <= page_num <= len(doc):
            raise ValueError(f"Page {page_num} hors du PDF ({len(doc)} pages)")
        page = doc[page_num - 1]

        # Recherche du texte "FIGURE X-Y" (sans le tiret pour être plus tolérant)
        anchor = f"FIGURE {figure_label}"
        bbox = find_figure_bbox(page, anchor)

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        if bbox is not None:
            pix = page.get_pixmap(matrix=matrix, clip=bbox)
            precise = True
        else:
            pix = page.get_pixmap(matrix=matrix)
            precise = False

        pix.save(str(out_path))
    return precise


# ---------------------------------------------------------------------------
# Appel VLM
# ---------------------------------------------------------------------------

def build_prompt(figure_label: str, caption: str, n_subs: int) -> str:
    if n_subs > 1:
        return (
            f"Cette image est la Figure {figure_label} d'un cours d'apprentissage "
            f"machine en français.\n"
            f"Légende : \u00ab {caption} \u00bb\n"
            f"Elle contient {n_subs} sous-figures organisées ensemble.\n\n"
            "Décris cette figure composite en 4 à 8 phrases directes en français :\n"
            "- ce que représente l'ensemble et l'idée pédagogique globale\n"
            "- ce que montre chaque sous-figure (a), (b), (c)... individuellement\n\n"
            "Réponds directement, sans phrase d'introduction du type "
            "'Cette figure montre'. Ne relève pas la légende."
        )
    return (
        f"Cette image est la Figure {figure_label} d'un cours d'apprentissage "
        f"machine en français.\n"
        f"Légende : \u00ab {caption} \u00bb\n\n"
        "Décris précisément en 2 à 4 phrases directes en français :\n"
        "- le type de figure (schéma, graphique, diagramme, table)\n"
        "- les éléments visibles\n"
        "- le concept pédagogique illustré\n\n"
        "Réponds directement, sans phrase d'introduction. Ne relève pas la légende."
    )


def transcribe_composite(image_path: Path, figure_label: str, caption: str,
                         n_subs: int, model: str) -> str:
    import ollama

    prompt = build_prompt(figure_label, caption, n_subs)
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [str(image_path)],
        }],
        options={"temperature": 0.2},
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Assemblage du .md final
# ---------------------------------------------------------------------------

def apply_replacements(md_content: str, groups_with_desc: list) -> str:
    """Remplace la 1re ref de chaque groupe par sa description; supprime les autres refs."""
    ops = []  # (start, end, replacement)
    for group, desc in groups_with_desc:
        first = group.sub_refs[0]
        block = f"**[Description de la Figure {group.label}]** {desc}"
        ops.append((first.start(), first.end(), block))
        for ref in group.sub_refs[1:]:
            ops.append((ref.start(), ref.end(), ""))

    ops.sort(key=lambda t: t[0], reverse=True)
    result = md_content
    for start, end, replacement in ops:
        result = result[:start] + replacement + result[end:]
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process(md_path: Path, pdf_path: Path, output_path: Path, model: str,
            dpi: int, dry_run: bool, save_crops_dir: Path,
            skip_vlm: bool):
    content = md_path.read_text(encoding="utf-8")
    groups = parse_figure_groups(content)

    print(f"[{md_path.name}] {len(groups)} figure(s) composite(s) détectée(s)")
    for g in groups:
        n = len(g.sub_refs)
        kind = "composite" if n > 1 else "simple"
        print(f"  Figure {g.label} : {n} sous-fig ({kind}), page {g.page}")
        print(f"    Légende : {g.caption[:80]}{'...' if len(g.caption) > 80 else ''}")

    if dry_run:
        return

    if save_crops_dir:
        save_crops_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp") / f"tf_{md_path.stem}"
    tmp_dir.mkdir(exist_ok=True)

    groups_with_desc = []
    for g in groups:
        n_subs = len(g.sub_refs)
        print(f"\n  Figure {g.label} (page {g.page}, {n_subs} sous-fig)...", flush=True)

        crop_name = f"figure_{g.label}.png"
        crop_path = tmp_dir / crop_name

        t0 = time.time()
        try:
            precise = crop_figure(pdf_path, g.page, g.label, crop_path, dpi=dpi)
        except Exception as e:
            print(f"    ERREUR crop : {e}")
            continue

        note = "" if precise else " (fallback page complète)"
        print(f"    Crop : {crop_path}{note}")

        if save_crops_dir:
            dest = save_crops_dir / crop_name
            shutil.copy(str(crop_path), str(dest))

        if skip_vlm:
            continue

        try:
            desc = transcribe_composite(crop_path, g.label, g.caption, n_subs, model)
        except Exception as e:
            print(f"    ERREUR VLM : {e}")
            continue

        dt = time.time() - t0
        print(f"    OK ({dt:.1f}s, {len(desc)} chars)")
        groups_with_desc.append((g, desc))

    if skip_vlm:
        print(f"\nCrops sauvés dans : {save_crops_dir or tmp_dir}")
        return

    new_content = apply_replacements(content, groups_with_desc)
    output_path.write_text(new_content, encoding="utf-8")
    print(f"\n-> {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="Transcription des figures composites d'un .md GLM-OCR."
    )
    p.add_argument("md_file", type=Path)
    p.add_argument("pdf_file", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("-m", "--model", default="qwen3-vl:8b")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--dry-run", action="store_true",
                   help="Liste les groupes détectés, ne crop rien.")
    p.add_argument("--save-crops", type=Path, default=None,
                   help="Sauvegarde les crops dans ce dossier (utile pour debug visuel).")
    p.add_argument("--skip-vlm", action="store_true",
                   help="Génère les crops seulement, sans appeler le VLM.")
    args = p.parse_args()

    if not args.md_file.exists():
        print(f"MD introuvable : {args.md_file}", file=sys.stderr)
        sys.exit(1)
    if not args.pdf_file.exists() and not args.dry_run:
        print(f"PDF introuvable : {args.pdf_file}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.md_file.with_suffix(".enriched.md")
    process(args.md_file, args.pdf_file, output, args.model,
            args.dpi, args.dry_run, args.save_crops, args.skip_vlm)


if __name__ == "__main__":
    main()