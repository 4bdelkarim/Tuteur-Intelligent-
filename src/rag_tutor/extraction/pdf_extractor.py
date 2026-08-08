#!/usr/bin/env python3
"""
extract_glmocr_layout.py — PDF (natif OU scanne) -> markdown unifie, via GLM-OCR SDK.

Pipeline :
  GLM-OCR SDK self-hosted  (PP-DocLayout-V3 sur CPU + OCR GLM-OCR via Ollama)
    -> JSON page-structure : regions + bbox normalisees 0-1000
  Post-traitement (ce fichier) :
    - regions texte/titre/formule/table  -> markdown tel quel (LaTeX propre de GLM-OCR)
    - regions FIGURE                     -> crop du bbox (PyMuPDF) -> VLM qwen2.5vl -> bloc [FIGURE]
  -> markdown unifie : front-matter + <!-- loc page=N --> + blocs [FIGURE] atomiques

Prerequis :
  pip install "glmocr[selfhosted]" pymupdf ollama
  # backend OCR : Ollama servant glm-ocr (endpoint OpenAI-compatible :11434/v1)
  # config.yaml : maas.enabled=false, enable_layout=true, layout.device=cpu, ocr_api->11434

  # 1) VERIFIER le schema JSON reel d'abord (indispensable) :
  python extract_glmocr_layout.py cours.pdf --config config.yaml --inspect
  # 2) puis produire le markdown :
  python extract_glmocr_layout.py cours.pdf --config config.yaml --out cours.md
  # 3) si un PDF precis a une page de garde/sommaire a exclure (AUCUNE page n'est
  #    sautee par defaut -> c'est un choix explicite, par-PDF, jamais automatique) :
  python extract_glmocr_layout.py cours.pdf --config config.yaml --skip-pages 1

IMPORTANT : le schema exact des regions GLM-OCR n'est pas documente publiquement.
Le mode --inspect imprime les cles reelles ; ajuste alors les listes *_KEYS / *_LABELS
ci-dessous si besoin (marquees TODO).
"""

import re
import json
import base64
import argparse
import unicodedata
from pathlib import Path

VLM_MODEL = "qwen3-vl:8b"
CROP_DPI  = 200
# AUCUNE page sautee par defaut. Sauter une page (sommaire/couverture) est une
# decision PROPRE A CHAQUE PDF, jamais une regle generale -> --skip-pages en CLI.
DEFAULT_SKIP_PAGES = set()

# --- Schema REEL du JSON GLM-OCR (confirme via --inspect) ---
TYPE_KEYS = ["label", "category", "type", "cls", "region_type"]     # type de region
BBOX_KEYS = ["bbox_2d", "bbox", "box", "polygon", "poly"]           # boite 0-1000
TEXT_KEYS = ["content", "markdown", "text", "md", "latex"]          # texte reconnu

FIGURE_LABELS  = {"image", "chart", "figure", "picture", "img"}     # regions figure
CAPTION_LABELS = {"figure_title", "table_title", "caption"}         # legendes / sous-legendes
TITLE_LABELS   = {"paragraph_title", "title", "section_title", "doc_title"}  # titres de section
DROP_LABELS    = {"header", "footer", "page_number", "page-number", "footnote_sep"}
# Legende PRINCIPALE (delimite une planche) : "FIGURE 3-4 ...", "TABLE .1 ..."
MAIN_CAPTION_RE = re.compile(r"^\s*(figure|table|tableau)\s+[\.\dIVX]", re.IGNORECASE)


# =====================================================
# Acces defensif aux champs (schema incertain)
# =====================================================

def _get(region, keys, default=None):
    for k in keys:
        if isinstance(region, dict) and k in region and region[k] not in (None, ""):
            return region[k]
    return default

def region_type(region):
    t = _get(region, TYPE_KEYS, "")
    return str(t).strip().lower()

def region_text(region):
    return str(_get(region, TEXT_KEYS, "") or "")

def region_bbox(region):
    """Renvoie (x1,y1,x2,y2) en 0-1000, ou None. Gere bbox rectangulaire ou polygone."""
    b = _get(region, BBOX_KEYS)
    if b is None:
        return None
    flat = []
    for v in (b if isinstance(b, (list, tuple)) else []):
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    nums = [float(x) for x in flat if isinstance(x, (int, float))]
    if len(nums) < 4:
        return None
    xs, ys = nums[0::2], nums[1::2]        # polygone -> enveloppe
    return (min(xs), min(ys), max(xs), max(ys))


# =====================================================
# Conversion bbox normalisee (0-1000) -> points PDF  (PUR, testable)
# =====================================================

def norm_bbox_to_points(bbox_norm, page_w_pt, page_h_pt):
    x1, y1, x2, y2 = bbox_norm
    return (x1 / 1000.0 * page_w_pt, y1 / 1000.0 * page_h_pt,
            x2 / 1000.0 * page_w_pt, y2 / 1000.0 * page_h_pt)


# =====================================================
# Crop d'une figure depuis le PDF (PyMuPDF)  [lazy fitz]
# =====================================================

def crop_region_png(doc, page_num, bbox_norm, dpi=CROP_DPI):
    import fitz
    page = doc[page_num - 1]
    r = page.rect
    x1, y1, x2, y2 = norm_bbox_to_points(bbox_norm, r.width, r.height)
    clip = fitz.Rect(x1, y1, x2, y2)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
    return pix.tobytes("png")


# =====================================================
# Description de figure par le VLM  [lazy ollama]
# =====================================================

# Prompt FIGURE SIMPLE (une seule image) — NE parle JAMAIS de sous-figures.
VLM_PROMPT_SIMPLE = (
    "Tu es un enseignant de Machine Learning qui explique une figure de cours a un etudiant.\n"
    "Voici UNE figure extraite d'un cours de Machine Learning.\n"
    "Legende (fait FOI) : {caption}\n\n"
    "Decris-la dans UN SEUL TEXTE FLUIDE (INTERDIT : titres, '###', listes a puces, numerotation) :\n"
    "d'abord ce qui est REELLEMENT visible (type de figure : schema, architecture de reseau, graphe de "
    "fonction, diagramme... ; axes et echelles reelles ; formes ; blocs ; fleches ; couleurs ; et les "
    "etiquettes exactes que tu lis), puis ce que la figure illustre sur le plan pedagogique, en restant "
    "STRICTEMENT rattache a ce que tu as observe.\n\n"
    "REGLES STRICTES :\n"
    "- Decris la figure comme UN TOUT coherent ; ne la decoupe PAS en morceaux artificiels.\n"
    "- N'invente AUCUNE valeur, echelle, couleur ou etiquette que tu ne vois pas.\n"
    "- Si un element est illisible, ecris-le explicitement plutot que de deviner.\n"
    "- Ne contredis jamais la legende. Reste factuel et concis. Reponds en francais."
)

# Prompt PLANCHE COMPOSITE (plusieurs images etiquetees) — traite chaque sous-figure.
VLM_PROMPT_COMPOSITE = (
    "Tu es un enseignant de Machine Learning qui explique une figure de cours a un etudiant.\n"
    "Cette figure est une planche composite ; les images ci-dessous sont ses sous-figures, dans cet ordre :\n"
    "{panneaux}\n"
    "Legende globale de la planche (fait FOI) : {caption}\n\n"
    "Analyse dans UN SEUL TEXTE FLUIDE (INTERDIT : titres, '###', listes a puces, numerotation) :\n"
    "d'abord, pour CHAQUE sous-figure, ce qui est REELLEMENT visible (axes et echelles reelles, forme reelle "
    "de la courbe : croissante / decroissante / en S / monotone / avec un pic..., etiquettes exactes lues), "
    "puis ce que la planche illustre sur le plan pedagogique.\n\n"
    "REGLES STRICTES :\n"
    "- Traite CHAQUE sous-figure separement et decris la forme PROPRE de sa courbe ; ne recopie pas la meme phrase d'une sous-figure a l'autre.\n"
    "- N'invente AUCUNE valeur, echelle (ex. ne dis pas 'logarithmique' si tu n'en es pas certain), couleur ou etiquette que tu ne vois pas.\n"
    "- Si tu ne reconnais pas une fonction, DECRIS SA FORME sans lui donner un nom errone (ne confonds pas, par ex., une tanh rectifiee |tanh(z)| avec une ReLU).\n"
    "- Si une courbe est illisible, ecris-le explicitement plutot que de deviner.\n"
    "- Ne contredis jamais la legende. Reste factuel et concis. Reponds en francais."
)


def describe_figure(images_png, labels, caption=""):
    """1 image  -> prompt SIMPLE (aucune mention de sous-figure).
       N images -> prompt COMPOSITE (sous-figures etiquetees, envoyees dans le meme appel)."""
    import ollama
    cap = caption or "(non disponible)"
    if len(images_png) > 1:
        lignes = "".join(f"  - Image {i}" + (f" = {lbl}" if lbl else "") + "\n"
                         for i, lbl in enumerate(labels, 1))
        prompt = VLM_PROMPT_COMPOSITE.replace("{panneaux}", lignes).replace("{caption}", cap)
    else:
        prompt = VLM_PROMPT_SIMPLE.replace("{caption}", cap)
    b64 = [base64.b64encode(p).decode() for p in images_png]
    r = ollama.chat(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt, "images": b64}],
        options={"temperature": 0},
    )
    return r["message"]["content"].strip()


# =====================================================
# Nettoyage commun leger (cf. Bloc 4)  (PUR)
# =====================================================

def tidy(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Prefixe de titre numerote : "1-", "1.1-", "3.10.2-.1"  -> profondeur = nb de niveaux
_HEADING_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)-(?:\.(\d+))?\s")

def format_heading(content):
    """Titre de section -> '#'*n selon la profondeur, en gardant le prefixe numerote."""
    content = content.strip()
    m = _HEADING_PREFIX.match(content)
    if not m:
        return f"## {content}"                       # titre sans prefixe (ex. SOMMAIRE)
    depth = m.group(1).count(".") + 1                 # 1- ->1 | 1.1- ->2 | 3.10.2- ->3
    if m.group(2):                                    # variante 3.10.2-.1  -> +1 niveau
        depth += 1
    level = min(depth + 1, 6)                         # niveau 1 -> ## ; plafond ######
    return f"{'#' * level} {content}"


# --- Conversion table HTML (sortie GLM-OCR) -> table Markdown (LaTeX des cellules preserve) ---
from html.parser import HTMLParser

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._in = [], None, None, False
    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell, self._in = [], True
    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row); self._row = None
        elif tag in ("td", "th") and self._row is not None:
            self._row.append("".join(self._cell).strip()); self._in = False
    def handle_data(self, data):
        if self._in:
            self._cell.append(data)

def html_table_to_md(html):
    """<table><tr><td>..</td></tr></table>  ->  table Markdown pipe.
       Les '|' des cellules (ex. |z| en LaTeX) sont echappes ; le LaTeX est garde tel quel."""
    p = _TableParser()
    try:
        p.feed(html)
    except Exception:
        return html                                   # parsing rate -> on garde le HTML
    rows = [r for r in p.rows if r]
    if not rows:
        return html
    ncol = max(len(r) for r in rows)
    def line(cells):
        cells = [c.replace("|", r"\|").replace("\n", " ") for c in cells]
        cells += [""] * (ncol - len(cells))
        return "| " + " | ".join(cells) + " |"
    md = [line(rows[0]), "| " + " | ".join(["---"] * ncol) + " |"]
    md += [line(r) for r in rows[1:]]
    return "\n".join(md)


# =====================================================
# Assemblage du markdown unifie  (PUR : injection VLM par callback)
# =====================================================

def _extract_pages(jr):
    """Le JSON du CLI peut etre une liste de pages, ou un dict qui la contient."""
    if isinstance(jr, list):
        return jr
    if isinstance(jr, dict):
        for k in ("pages", "json_result", "result", "data"):
            if isinstance(jr.get(k), list):
                return jr[k]
        for v in jr.values():                 # sinon : premiere valeur liste
            if isinstance(v, list):
                return v
    return []


def _envelope(regions):
    """Boite englobante (0-1000) d'un groupe de regions."""
    boxes = [b for b in (region_bbox(r) for r in regions) if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _pair_subfigures(figs, subs):
    """Associe chaque figure a la sous-legende la plus proche EN DESSOUS (recouvrement horizontal).
       Renvoie une liste ordonnee de (bbox, sous_legende)."""
    panels, used = [], set()
    for f in sorted(figs, key=lambda r: region_bbox(r)[1] if region_bbox(r) else 0):
        fb = region_bbox(f)
        if not fb:
            continue
        best, best_gap = None, 1e9
        for i, s in enumerate(subs):
            if i in used:
                continue
            sb = region_bbox(s)
            if not sb:
                continue
            x_overlap = min(fb[2], sb[2]) - max(fb[0], sb[0])   # recouvrement horizontal
            gap = sb[1] - fb[3]                                  # ecart vertical (sous-leg sous la figure)
            if x_overlap > 0 and -10 <= gap < best_gap:
                best, best_gap = i, gap
        label = region_text(subs[best]).strip() if best is not None else ""
        if best is not None:
            used.add(best)
        panels.append((fb, label))
    return panels


def _emit_figure(out, figs, subs, caption, pnum, describe_fn):
    """AIGUILLAGE :
       - sous-legendes (a)(b)... presentes -> planche COMPOSITE : chaque sous-figure croppee
         separement + etiquetee -> N images + prompt composite.
       - pas de sous-legende -> figure SIMPLE : UNE image (enveloppe de toutes les regions,
         jamais fragmentee) + prompt simple.
    """
    if len(subs) >= 2:                                  # planche composite
        panels = [(b, l) for b, l in _pair_subfigures(figs, subs) if b]
    else:                                               # figure simple -> une seule image
        env = _envelope(figs)
        panels = [(env, "")] if env else []
    if not panels:
        return
    desc = describe_fn(pnum, panels, caption)
    out.append("")                          # bloc isole
    out.append("--- [FIGURE] ---")
    if caption:
        out.append(caption)                 # on garde la VRAIE legende (ancrage)
    out.append(desc)
    out.append("--- [/FIGURE] ---")
    out.append("")


def build_markdown(json_result, source_id, describe_fn, skip_pages=None):
    """describe_fn(page_num, bbox_norm, caption) -> description figure.
    Regroupe les sous-figures par legende principale (FIGURE N -) avant de cropper.
    describe_fn injectable -> testable sans VLM ni PDF reels.

    skip_pages : ensemble de numeros de page (1-indexes) a EXCLURE. Par defaut
    (None) -> AUCUNE page n'est sautee (DEFAULT_SKIP_PAGES = set()). C'est un
    choix explicite par PDF (--skip-pages en CLI), jamais une regle automatique."""
    skip = DEFAULT_SKIP_PAGES if skip_pages is None else skip_pages
    pages = _extract_pages(json_result)
    out = ["---", "source_type: pdf", f"source_id: {source_id}",
           f"page_count: {len(pages)}", "---", ""]

    for pnum, page in enumerate(pages, start=1):
        if pnum in skip:
            continue
        regions = page if isinstance(page, list) else page.get("regions", page.get("blocks", []))
        regions = sorted(regions, key=lambda r: r.get("index", 0) if isinstance(r, dict) else 0)
        out.append(f"<!-- loc page={pnum} -->")

        figs, subs = [], []                 # planche en cours d'accumulation
        def flush(caption=""):
            nonlocal figs, subs
            if figs:
                _emit_figure(out, figs, subs, caption, pnum, describe_fn)
            figs, subs = [], []

        for reg in regions:
            lab = region_type(reg)
            if lab in DROP_LABELS:
                continue
            if lab in FIGURE_LABELS:
                figs.append(reg)
            elif lab in CAPTION_LABELS:
                content = region_text(reg).strip()
                if MAIN_CAPTION_RE.match(content):
                    flush(content)          # legende principale -> ferme la planche
                else:
                    subs.append(reg)        # sous-legende (a)(b)... -> dans la planche
            else:                            # titre / texte / formule / table / algorithme
                content = region_text(reg).strip()
                if not content:
                    continue
                flush()                      # securite d'ordre si figures en attente
                if lab in TITLE_LABELS:
                    out.append("")
                    out.append(format_heading(content))
                    out.append("")
                elif lab == "table" or content.lstrip().startswith("<table"):
                    out.append("")
                    out.append(html_table_to_md(content))   # HTML -> markdown (LaTeX preserve)
                    out.append("")
                else:
                    out.append(content)
                    out.append("")           # paragraphes aeres (ligne vide entre chacun)
        flush()                              # fin de page : planche sans legende explicite
        out.append("")

    return tidy("\n".join(out))


# =====================================================
# Execution GLM-OCR : on appelle le CLI VERIFIE (pas l'API Python qui retombe sur MaaS)
# =====================================================

def run_glmocr(pdf_path, config_path, out_dir="_glmocr_json"):
    """Lance la commande CLI qui marche chez toi, puis charge le JSON produit.
       Equivaut a :  glmocr parse <pdf> --config <cfg> --mode selfhosted --layout-device cpu --output <dir>
    """
    import subprocess
    import glob
    out = Path(out_dir); out.mkdir(exist_ok=True)
    cmd = ["glmocr", "parse", str(pdf_path), "--config", str(config_path),
           "--mode", "selfhosted", "--layout-device", "cpu", "--output", str(out)]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)

    stem = Path(pdf_path).stem
    jsons = sorted(glob.glob(str(out / "**" / "*.json"), recursive=True),
                   key=lambda p: Path(p).stat().st_mtime)
    if not jsons:
        raise FileNotFoundError(
            f"Aucun .json produit dans {out}/. Verifie ce que le CLI ecrit "
            f"(regarde le dossier de sortie de ta commande qui marche)."
        )
    match = [j for j in jsons if stem in Path(j).stem]
    chosen = match[-1] if match else jsons[-1]
    print(f"JSON charge : {chosen}")
    return json.loads(Path(chosen).read_text(encoding="utf-8"))

def process(pdf_path, config_path, out_path, skip_pages=None):
    import fitz
    import time
    json_result = run_glmocr(pdf_path, config_path)
    doc = fitz.open(str(pdf_path))

    counter = {"n": 0}

    def describe(pnum, panels, caption):
        # Boucle VLM SILENCIEUSE avant ce patch -> ressemblait a un blocage.
        # Chaque figure imprime maintenant sa progression + son temps de reponse.
        counter["n"] += 1
        cap_short = (caption or "(sans legende)")[:55]
        print(f"  [figure {counter['n']}] page {pnum} ({len(panels)} panneau(x)) "
              f"— {cap_short!r} — VLM en cours...", flush=True)
        t0 = time.time()
        images = [crop_region_png(doc, pnum, bbox) for bbox, _ in panels]
        labels = [lbl for _, lbl in panels]
        desc = describe_figure(images, labels, caption)
        print(f"    -> termine en {time.time() - t0:.1f}s", flush=True)
        return desc

    md = build_markdown(json_result, Path(pdf_path).name, describe, skip_pages=skip_pages)
    Path(out_path).write_text(md, encoding="utf-8")
    doc.close()
    print(f"OK -> {out_path}  ({len(md)} caracteres, {counter['n']} figure(s) decrite(s))")


# =====================================================
# Mode --inspect : imprime le schema reel des regions
# =====================================================

def inspect(pdf_path, config_path):
    jr = run_glmocr(pdf_path, config_path)
    pages = _extract_pages(jr)
    print(f"\nPages: {len(pages)}")
    if not pages:
        print("Aucune page reconnue — structure JSON brute :", type(jr).__name__, str(jr)[:300]); return
    p0 = pages[0]
    regions = p0 if isinstance(p0, list) else p0.get("regions", p0.get("blocks", []))
    print(f"Regions page 1: {len(regions)}")
    if regions:
        r0 = regions[0]
        print("Cles d'une region :", list(r0.keys()) if isinstance(r0, dict) else type(r0).__name__)
        print("Exemple region[0] :", json.dumps(r0, ensure_ascii=False)[:400])
        print("\n-> Ajuste TYPE_KEYS / BBOX_KEYS / TEXT_KEYS / FIGURE_LABELS si les cles different.")


def _parse_skip_pages(spec):
    """'1' | '1,3,5' | '' -> set d'entiers. Vide/None -> set() (rien saute)."""
    if not spec:
        return set()
    return {int(p.strip()) for p in spec.split(",") if p.strip()}


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GLM-OCR (layout) -> markdown unifie.")
    ap.add_argument("pdf", nargs="+")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true", help="imprime le schema JSON reel puis quitte")
    ap.add_argument("--skip-pages", default="",
                    help="pages a exclure (1-indexees), ex. '1' ou '1,3' — AUCUNE par defaut, "
                         "propre a CE pdf (pas une regle globale)")
    args = ap.parse_args()

    skip_pages = _parse_skip_pages(args.skip_pages)

    if args.inspect:
        inspect(args.pdf[0], args.config)
    else:
        for pdf in args.pdf:
            if args.out:
                out = Path(args.out) / (Path(pdf).stem + ".md")
            else:
                out = Path(pdf).with_suffix(".md")
            process(pdf, args.config, str(out), skip_pages=skip_pages)