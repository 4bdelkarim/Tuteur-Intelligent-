#!/usr/bin/env python3
"""
extract_glmocr_layout.py — PDF (natif OU scanne) -> markdown unifie, via GLM-OCR SDK.

Pipeline :
  1. OCR PAGE ENTIERE (1 appel "Text Recognition:" par page, GLM-OCR via Ollama)
     -> texte + formules + tableaux PROPRES. Le mode par-region du SDK glmocr croppe
        chaque region et fait un appel par region -> sur des crops minuscules, glm-ocr
        boucle / "raisonne a voix haute" / emet des fences vides (abandonne).
  2. Layout PP-DocLayoutV3 (utilise UNIQUEMENT pour les FIGURES) :
     -> detection + crop des figures (image/chart) + OCR de leurs legendes.
  3. Post-traitement (ce fichier) :
     - texte page entiere -> markdown (LaTeX propre, bruit nettoye)
     - figures -> crop + legende -> VLM qwen3-vl:8b -> bloc [FIGURE]
  -> markdown unifie : front-matter + <!-- loc page=N --> + blocs [FIGURE] atomiques

Prerequis :
  pip install "glmocr[selfhosted]" ollama   # pymupdf optionnel (fallback crop seulement)
  # backend OCR : Ollama servant glm-ocr sur 127.0.0.1:11434 (config.yaml, api_mode=ollama_generate)
  # le device du layout est pilote par config.yaml (pipeline.layout.device), pas force en CLI

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

import io
import re
import json
import base64
import argparse
import unicodedata
from pathlib import Path

VLM_MODEL = "qwen3-vl:8b"
CROP_DPI  = 200
# qwen3-vl:8b pese ~45 Go : le CHARGEMENT (quand il n'est pas deja en VRAM)
# prend ~5 min et depassait l'ancien timeout -> ressemblait a un blocage.
# On le charge UNE fois explicitement au premier appel VLM (warm-up).
_WARMED = {"done": False}
# AUCUNE page sautee par defaut. Sauter une page (sommaire/couverture) est une
# decision PROPRE A CHAQUE PDF, jamais une regle generale -> --skip-pages en CLI.
DEFAULT_SKIP_PAGES = set()

# --- Schema REEL du JSON GLM-OCR (confirme via --inspect) ---
# Le type BRUT est dans `label` (text/image/formula/table). L'information
# SEMANTIQUE fine (figure_title, paragraph_title, chart, display_formula,
# algorithm...) est dans `native_label` -> a lire EN PRIORITE.
NATIVE_LABEL_KEYS = ["native_label", "native_type", "sub_label", "sub_type"]
TYPE_KEYS = ["label", "category", "type", "cls", "region_type"]     # type brut (fallback)
BBOX_KEYS = ["bbox_2d", "bbox", "box", "polygon", "poly"]           # boite 0-1000
TEXT_KEYS = ["content", "markdown", "text", "md", "latex"]          # texte reconnu
IMAGE_PATH_KEYS = ["image_path", "img_path", "image", "crop_path"]  # crop figure deja fait

FIGURE_LABELS  = {"image", "chart", "figure", "picture", "img"}     # regions figure
CAPTION_LABELS = {"figure_title", "table_title", "caption"}         # legendes / sous-legendes
TITLE_LABELS   = {"paragraph_title", "title", "section_title", "doc_title"}  # titres de section
DROP_LABELS    = {"header", "footer", "page_number", "page-number", "footnote_sep"}
# Legende PRINCIPALE (delimite une planche) : "FIGURE 3-4 ...", "TABLE .1 ..."
MAIN_CAPTION_RE = re.compile(r"^\s*(figure|table|tableau)\s+[\.\dIVX]", re.IGNORECASE)
# Legende de TABLEAU (pas une figure) : "TABLE .1", "Tableau 2" -> a garder comme texte
TABLE_CAPTION_RE = re.compile(r"^\s*(table|tableau)\s+[\.\dIVX]", re.IGNORECASE)
# Sous-legende d'une planche composite : "(a) ...", "(b) ..."
SUB_CAPTION_RE = re.compile(r"^\s*\([a-z]\)", re.IGNORECASE)


# =====================================================
# Acces defensif aux champs (schema incertain)
# =====================================================

def _get(region, keys, default=None):
    for k in keys:
        if isinstance(region, dict) and k in region and region[k] not in (None, ""):
            return region[k]
    return default

def region_type(region):
    """Type SEMANTIQUE de la region : `native_label` d'abord (figure_title,
    paragraph_title, chart, display_formula, algorithm...), puis fallback sur
    le type brut `label` (text/image/formula/table)."""
    t = _get(region, NATIVE_LABEL_KEYS, "") or _get(region, TYPE_KEYS, "")
    return str(t).strip().lower()

def region_text(region):
    return str(_get(region, TEXT_KEYS, "") or "")

def region_image_path(region):
    """Chemin (relatif au dossier du JSON) de l'image deja croppee par GLM-OCR, ou None."""
    p = _get(region, IMAGE_PATH_KEYS, None)
    return str(p).strip() if p else None

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
       N images -> prompt COMPOSITE (sous-figures etiquetees, envoyees dans le meme appel).
       Timeout + 1 retry : un appel VLM qui bloque ne doit pas faire tomber tout le pipeline."""
    import ollama
    cap = caption or "(non disponible)"
    if len(images_png) > 1:
        lignes = "".join(f"  - Image {i}" + (f" = {lbl}" if lbl else "") + "\n"
                         for i, lbl in enumerate(labels, 1))
        prompt = VLM_PROMPT_COMPOSITE.replace("{panneaux}", lignes).replace("{caption}", cap)
    else:
        prompt = VLM_PROMPT_SIMPLE.replace("{caption}", cap)
    b64 = [base64.b64encode(p).decode() for p in images_png]
    client = ollama.Client(timeout=600)
    if not _WARMED["done"]:
        import time
        print("  [VLM] chargement du modele qwen3-vl:8b (premiere fois, ~5 min)...", flush=True)
        t0 = time.time()
        try:
            client.chat(model=VLM_MODEL,
                        messages=[{"role": "user", "content": "OK"}],
                        options={"temperature": 0})
        except Exception as e:
            print(f"    ⚠️ warm-up VLM: {e}", flush=True)
        _WARMED["done"] = True
        print(f"    -> modele pret en {time.time() - t0:.1f}s", flush=True)
    for attempt in (1, 2):
        try:
            r = client.chat(
                model=VLM_MODEL,
                messages=[{"role": "user", "content": prompt, "images": b64}],
                options={"temperature": 0},
            )
            return r["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                print(f"    ⚠️ VLM echoue apres 2 essais ({e}) — description vide", flush=True)
                return ""
            print(f"    ⚠️ VLM timeout (essai {attempt}) — nouvelle tentative...", flush=True)


# =====================================================
# Nettoyage commun leger (cf. Bloc 4)  (PUR)
# =====================================================

def tidy(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =====================================================
# Nettoyage du contenu GLM-OCR (bruit ```markdown + doublons)
# =====================================================

def _cut_fence(raw):
    """GLM-OCR repete le contenu dans un bloc ```markdown : on coupe au premier fence."""
    return re.split(r"```", raw, maxsplit=1)[0].strip()

def _first_formula(raw):
    m = re.search(r"\$\$.*?\$\$", raw, re.DOTALL)
    return m.group(0).strip() if m else raw

def _first_table(raw):
    m = re.search(r"<table\b.*?</table>", raw, re.DOTALL | re.IGNORECASE)
    return m.group(0).strip() if m else raw

def _dedupe_lines(raw):
    """Supprime les lignes consecutives strictement identiques (boucle du modele)
    ainsi qu'une ligne finale tronquee (repetition coupee par num_predict)."""
    out = []
    for line in raw.splitlines():
        if out and line.strip() == out[-1].strip():
            continue
        # ligne plus courte = prefixe de la precedente -> repetition tronquee
        if out and line.strip() and out[-1].strip().startswith(line.strip()):
            continue
        out.append(line)
    return "\n".join(out)

def clean_content(lab, raw):
    """Retourne le contenu propre d'une region selon son type semantique.

    GLM-OCR repete le contenu dans des blocs ```markdown, duplique formules et
    tables, et laisse parfois fuiter ses instructions OCR dans les titres ;
    on garde l'occurrence canonique (la premiere)."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if lab in ("display_formula", "inline_formula"):
        return _first_formula(_cut_fence(raw))
    if lab == "table" or raw.lstrip().startswith("<table"):
        return _first_table(raw)
    if lab == "algorithm":
        # le texte markdown est la forme canonique ; la table HTML qui suit est un doublon
        return _cut_fence(raw.split("<table", 1)[0])
    if lab in TITLE_LABELS:
        # un titre = sa PREMIERE ligne (le reste est doublon ou fuite d'instructions OCR)
        return _cut_fence(raw).splitlines()[0].strip()
    if lab in CAPTION_LABELS:
        # une legende = sa premiere occurrence (le modele repete la ligne, cf. OCR page entiere)
        return _dedupe_lines(_cut_fence(raw)).strip()
    return _cut_fence(raw)


# Prefixe de titre numerote : "1-", "1.1-", "3.10.2-.1"  -> profondeur = nb de niveaux
_HEADING_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)-(?:\.(\d+))?\s")

def format_heading(content):
    """Titre de section -> '#'*n selon la profondeur, en gardant le prefixe numerote."""
    content = re.sub(r"^#{1,6}\s*", "", content.strip())   # GLM-OCR prefixe deja en ##
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


def _pair_subfigures(figs, subs):
    """Associe chaque figure a la sous-legende la plus proche EN DESSOUS (recouvrement horizontal).
       Renvoie une liste ordonnee de (region, sous_legende).

       L'ordre de lecture gauche->droite est celui du champ `index` de GLM-OCR
       (les regions arrivent deja triees par index dans build_markdown). Un tri
       par bbox seul reordonne mal les figures d'une meme ligne (ex. FIGURE 4-9)."""
    panels, used = [], set()
    for f in figs:
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
        label = clean_content("figure_title", region_text(subs[best])) if best is not None else ""
        if best is not None:
            used.add(best)
        panels.append((f, label))
    return panels


def _figure_description(pnum, figs, subs, caption, describe_fn):
    """AIGUILLAGE simple vs composite -> description VLM (None si aucune figure).
       describe_fn(pnum, panels, caption) avec panels = liste de (region, label)."""
    if len(subs) >= 2:                                  # planche composite
        panels = [(r, l) for r, l in _pair_subfigures(figs, subs) if r]
    else:                                               # figure simple -> crop(s) GLM-OCR
        panels = [(r, "") for r in figs]
    if not panels:
        return None
    return describe_fn(pnum, panels, caption)

def _emit_figure(out, figs, subs, caption, pnum, describe_fn):
    """Ecrit le bloc [FIGURE] (legende + description) dans `out` (liste de lignes)."""
    desc = _figure_description(pnum, figs, subs, caption, describe_fn)
    if desc is None:
        return
    out.append("")                          # bloc isole
    out.append("--- [FIGURE] ---")
    if caption:
        out.append(caption)                 # on garde la VRAIE legende (ancrage)
    out.append(desc)
    out.append("--- [/FIGURE] ---")
    out.append("")


def build_markdown(json_result, source_id, describe_fn, skip_pages=None):
    """describe_fn(page_num, panels, caption) -> description figure.
    panels = liste de (region, sous_legende) ; chaque region porte son image_path
    (crop deja fait par GLM-OCR). Regroupe les sous-figures par legende principale.
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
                content = clean_content(lab, region_text(reg))
                if TABLE_CAPTION_RE.match(content):
                    # legende de TABLEAU -> texte (elle suit la table deja emise)
                    flush()
                    out.append(content)
                    out.append("")
                elif MAIN_CAPTION_RE.match(content):
                    flush(content)          # legende de figure -> ferme la planche
                else:
                    subs.append(reg)        # sous-legende (a)(b)... -> dans la planche
            else:                            # titre / texte / formule / table / algorithme
                content = clean_content(lab, region_text(reg))
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
# Assemblage markdown depuis l'OCR PAGE ENTIERE  (PUR : injection VLM par callback)
# =====================================================

def _norm(s):
    """Normalise pour comparer des chaines malgre la ponctuation (ex. - vs –)."""
    return re.sub(r"[^\w]+", "", (s or "").lower())

def _find_caption_line(text, caption):
    """Indice de la ligne du texte correspondant a la legende (None si absente)."""
    if not caption:
        return None
    nc = _norm(caption)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _norm(line) == nc:
            return i
    for i, line in enumerate(lines):
        nl = _norm(line)
        if nc and nl and (nc in nl or nl in nc):
            return i
    return None

def _cluster_figures(figures):
    """Groupe les regions figure en planches : des sous-figures proches
    verticalement (meme bande) -> UNE planche composite ; sinon figure simple."""
    figs = sorted(figures, key=lambda r: (r["bbox_2d"][1], r["bbox_2d"][0]))
    clusters = []
    for f in figs:
        y1, y2 = f["bbox_2d"][1], f["bbox_2d"][3]
        for c in clusters:
            cy1 = min(r["bbox_2d"][1] for r in c)
            cy2 = max(r["bbox_2d"][3] for r in c)
            if y1 < cy2 + 40 and y2 > cy1 - 40:      # bande verticale commune / proche
                c.append(f)
                break
        else:
            clusters.append([f])
    return clusters

def _extract_captions(text):
    """Legendes principales (FIGURE ...) du texte page entiere, dans l'ordre de
    lecture. Les legendes de TABLEAU sont exclues (elles restent du texte)."""
    caps = []
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        if MAIN_CAPTION_RE.match(s) and not TABLE_CAPTION_RE.match(s):
            caps.append((s, i))
    return caps

def _subcaptions_for_cluster(cluster, sub_caps):
    """Sous-legendes dont la bbox est dans (ou juste sous) la bande du cluster."""
    cy1 = min(r["bbox_2d"][1] for r in cluster)
    cy2 = max(r["bbox_2d"][3] for r in cluster)
    out = []
    for c in sub_caps:
        y1, y2 = c["bbox_2d"][1], c["bbox_2d"][3]
        if cy1 - 10 <= y1 <= cy2 + 80:
            out.append(c)
    return out

def _build_figure_blocks(pnum, page, describe_fn):
    """Associe chaque planche de figures a sa legende et a ses sous-legendes :
    -> liste de (legende, description).
    - Legende principale : TEXTE page entiere (deterministe, LaTeX exact),
      fallback OCR.
    - Sous-legendes (a)(b)... : OCR des regions figure_title (crop + contexte),
      puis appariement a chaque sous-figure via _pair_subfigures -> etiquettes
      du prompt COMPOSITE."""
    clusters = _cluster_figures(page["figures"])
    text_captions = _extract_captions(page["text"])
    ocr_captions = page.get("captions", [])
    sub_caps = [c for c in ocr_captions
                if SUB_CAPTION_RE.match(clean_content("figure_title", c["content"]))]
    main_caps = [c["content"] for c in ocr_captions
                 if MAIN_CAPTION_RE.match(clean_content("figure_title", c["content"]))]
    blocks = []
    for k, cluster in enumerate(clusters):
        if k < len(text_captions):
            caption = text_captions[k][0]
        elif k < len(main_caps):
            caption = main_caps[k]
        else:
            caption = ""
        subs = _subcaptions_for_cluster(cluster, sub_caps)
        desc = _figure_description(pnum, cluster, subs, caption, describe_fn)
        if desc is not None:
            blocks.append((caption, desc))
    return blocks

def _insert_figure(text, caption, desc):
    """Insere le bloc [FIGURE] dans le texte de la page.

    La legende est retrouvee dans le texte (normalise) -> on REPLACE sa ligne par
    le bloc complet (legende LaTeX EXACTE conservee). Sinon (legende absente du
    texte, ex. planche en tete de page), le bloc est mis en tete de page."""
    lines = text.splitlines()
    idx = _find_caption_line(text, caption)
    if idx is not None:
        cap = lines[idx]                       # legende originale (LaTeX exact)
        lines[idx] = f"--- [FIGURE] ---\n{cap}\n{desc}\n--- [/FIGURE] ---"
        return "\n".join(lines)
    cap = caption or ""
    block = f"--- [FIGURE] ---\n{cap}\n{desc}\n--- [/FIGURE] ---"
    return block + "\n\n" + text.lstrip()

def build_markdown_fullpage(result, source_id, describe_fn, skip_pages=None):
    """Assemble le markdown final depuis l'OCR page entiere.

    describe_fn(page_num, panels, caption) -> description figure, injectable et
    testable (meme contrat que build_markdown). Le texte vient de l'OCR page
    entiere (propre) ; les figures viennent du layout, les legendes du texte."""
    skip = DEFAULT_SKIP_PAGES if skip_pages is None else skip_pages
    pages = result["pages"]
    out = ["---", "source_type: pdf", f"source_id: {source_id}",
           f"page_count: {len(pages)}", "---", ""]
    for pnum, page in enumerate(pages, start=1):
        if pnum in skip:
            continue
        out.append(f"<!-- loc page={pnum} -->")
        out.append("")
        body = (page["text"] or "").strip()
        for caption, desc in _build_figure_blocks(pnum, page, describe_fn):
            body = _insert_figure(body, caption, desc)
        out.append(body)
        out.append("")
    return tidy("\n".join(out))


# =====================================================
# Execution GLM-OCR : OCR PAGE ENTIERE (1 appel "Text Recognition:" / page)
# + layout PP-DocLayoutV3 pour DETECTER/CROPPER les figures (legendes -> texte).
# =====================================================

def _ollama_ocr(pil_img, model, host, port, prompt, num_predict=4096):
    """Un appel OCR Ollama (/api/generate) sur une image PIL -> texte."""
    import urllib.request
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.0, "top_k": 1},
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())["response"]

def _crop_pil(img, bbox, pad=False):
    """Crop d'une bbox normalisee 0-1000 -> PIL.Image (pixels)."""
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = x1 / 1000 * W, x2 / 1000 * W
    y1, y2 = y1 / 1000 * H, y2 / 1000 * H
    if pad:  # les legendes sont fines -> marge pour donner du contexte au modele
        x1, x2 = max(0, x1 - (x2 - x1) * 0.05), min(W, x2 + (x2 - x1) * 0.05)
        y1, y2 = max(0, y1 - max((y2 - y1) * 0.4, 25)), min(H, y2 + max((y2 - y1) * 0.4, 25))
    return img.crop((int(x1), int(y1), int(x2), int(y2)))

def _caption_context_bbox(caption_bbox, figure_bboxes):
    """Etend la bbox d'une legende vers le HAUT pour inclure la figure situee juste
    au-dessus (contexte -> l'OCR lit la legende au lieu de boucler)."""
    cx1, cy1, cx2, cy2 = caption_bbox
    best, best_gap = None, 1e9
    for fx1, fy1, fx2, fy2 in figure_bboxes:
        x_overlap = min(cx2, fx2) - max(cx1, fx1)
        gap = cy1 - fy2                       # figure au-dessus de la legende
        if x_overlap > 0 and 0 <= gap < best_gap:
            best, best_gap = (fx1, fy1, fx2, fy2), gap
    if best is not None:
        fx1, fy1, fx2, fy2 = best
        return (min(cx1, fx1), fy1, max(cx2, fx2), cy2)
    return (cx1, cy1 - 30, cx2, cy2 + 15)

def _extract_caption_line(ocr_raw):
    """Premiere ligne qui ressemble a une legende (sub '(a)' ou principale 'FIGURE')."""
    for line in ocr_raw.splitlines():
        s = line.strip()
        if SUB_CAPTION_RE.match(s) or MAIN_CAPTION_RE.match(s):
            return s
    return ""

def _ocr_caption_robust(img, bbox, figure_bboxes, model, host, port, prompt_text):
    """OCR d'une legende : crop avec marge, puis retry sur un crop incluant la
    figure au-dessus si la legende n'est pas reconnue (le petit crop renvoie
    parfois des fences vides)."""
    cap = _extract_caption_line(_ollama_ocr(_crop_pil(img, bbox, pad=True),
                                            model, host, port, prompt_text, 512))
    if cap:
        return cap
    ctx = _caption_context_bbox(bbox, figure_bboxes)
    return _extract_caption_line(_ollama_ocr(_crop_pil(img, ctx),
                                             model, host, port, prompt_text, 1024))

def run_glmocr_fullpage(pdf_path, config_path, out_dir="_glmocr_json"):
    """OCR PAGE ENTIERE + layout pour les figures. Renvoie (result, base_dir).

    Pourquoi ne PAS utiliser le mode par-region du SDK : il croppe chaque region
    et fait un appel OCR par region -> sur des crops minuscules, glm-ocr boucle /
    "raisonne a voix haute" / emet des fences vides. En page entiere, le modele
    produit un texte EXACT et propre (cf. rapport 5.3.1).    Le layout n'est plus
    utilise que pour DETECTER et CROPPER les figures (les legendes sont lues dans
    le texte page entiere, pas OCRees a part -> fiable).

    result = {"pages": [{"text", "figures": [region], "captions": [region]}, ...]}
    base_dir = dossier des images croppees (resout les image_path)."""
    import fitz
    from PIL import Image

    from glmocr.config import GlmOcrConfig
    cfg = GlmOcrConfig.from_yaml(config_path)
    api = cfg.pipeline.ocr_api
    model = api.model or "glm-ocr:latest"
    host = api.api_host or "localhost"
    port = api.api_port or 11434
    dpi = cfg.pipeline.page_loader.pdf_dpi or 200
    prompt_text = (cfg.pipeline.page_loader.task_prompt_mapping or {}).get("text") \
        or "Text Recognition:"

    # --- rendu des pages ---
    doc = fitz.open(str(pdf_path))
    try:
        page_imgs = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            page_imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    finally:
        doc.close()

    # --- layout (figures + legendes seulement) ---
    from glmocr.layout import PPDocLayoutDetector
    detector = PPDocLayoutDetector(cfg.pipeline.layout)
    detector.start()
    try:
        layout_results, _vis = detector.process(page_imgs)
    finally:
        detector.stop()

    out = Path(out_dir)
    (out / "imgs").mkdir(parents=True, exist_ok=True)

    pages = []
    for pnum, (img, layout) in enumerate(zip(page_imgs, layout_results), start=1):
        text = _ollama_ocr(img, model, host, port, prompt_text, num_predict=4096)
        figs, cap_regions = [], []
        for reg in sorted(layout, key=lambda r: r.get("index", 0)):
            label = str(reg.get("label", "")).strip().lower()
            bbox = reg.get("bbox_2d")
            if not bbox:
                continue
            if label in FIGURE_LABELS:
                crop = _crop_pil(img, bbox)
                name = f"cropped_page{pnum}_idx{reg.get('index', 0)}.jpg"
                crop.save(out / "imgs" / name, quality=92)
                figs.append({"label": label, "bbox_2d": bbox, "image_path": f"imgs/{name}"})
            elif label in CAPTION_LABELS:
                cap_regions.append({"bbox_2d": bbox})
        # OCR des legendes APRES les figures (pour connaitre leurs bbox en contexte)
        fig_bboxes = [f["bbox_2d"] for f in figs]
        captions = []
        for c in cap_regions:
            content = _ocr_caption_robust(img, c["bbox_2d"], fig_bboxes,
                                          model, host, port, prompt_text)
            if content:
                captions.append({"bbox_2d": c["bbox_2d"], "content": content})
                print(f"    legende p{pnum} {c['bbox_2d']}: {content!r}", flush=True)
        pages.append({"text": text, "figures": figs, "captions": captions})
        print(f"  page {pnum}: OCR {len(text)} chars, {len(figs)} figure(s), "
              f"{len(captions)} legende(s)", flush=True)

    return {"pages": pages}, out

def process(pdf_path, config_path, out_path, skip_pages=None):
    import time
    result, base_dir = run_glmocr_fullpage(pdf_path, config_path)

    counter = {"n": 0}
    doc_holder = {"handle": None}          # fitz ouvert paresseusement (fallback seulement)

    def _crop_fallback(pnum, bbox):
        if doc_holder["handle"] is None:
            import fitz
            doc_holder["handle"] = fitz.open(str(pdf_path))
        return crop_region_png(doc_holder["handle"], pnum, bbox)

    def describe(pnum, panels, caption):
        # Boucle VLM SILENCIEUSE avant ce patch -> ressemblait a un blocage.
        # Chaque figure imprime maintenant sa progression + son temps de reponse.
        counter["n"] += 1
        cap_short = (caption or "(sans legende)")[:55]
        print(f"  [figure {counter['n']}] page {pnum} ({len(panels)} panneau(x)) "
              f"— {cap_short!r} — VLM en cours...", flush=True)
        t0 = time.time()
        images, labels = [], []
        for reg, lbl in panels:
            img_path = region_image_path(reg)
            p = (base_dir / img_path).resolve() if img_path else None
            if p and p.exists():
                images.append(p.read_bytes())       # crop GLM-OCR (pas de re-crop)
                labels.append(lbl)
            else:                                    # fallback : crop PyMuPDF
                bbox = region_bbox(reg)
                if bbox:
                    images.append(_crop_fallback(pnum, bbox))
                    labels.append(lbl)
        desc = describe_figure(images, labels, caption) if images else ""
        print(f"    -> termine en {time.time() - t0:.1f}s", flush=True)
        return desc

    md = build_markdown_fullpage(result, Path(pdf_path).name, describe, skip_pages=skip_pages)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(md, encoding="utf-8")
    if doc_holder["handle"] is not None:
        doc_holder["handle"].close()
    print(f"OK -> {out_path}  ({len(md)} caracteres, {counter['n']} figure(s) decrite(s))")


# =====================================================
# Mode --inspect : imprime le schema reel des regions
# =====================================================

def inspect(pdf_path, config_path):
    result, _base = run_glmocr_fullpage(pdf_path, config_path)
    pages = result["pages"]
    print(f"\nPages: {len(pages)}")
    if pages:
        p0 = pages[0]
        print("Texte page 1 (extrait) :", repr(p0["text"][:200]))
        print(f"Figures page 1 : {len(p0['figures'])}")
        if p0["figures"]:
            print("Exemple figure :", json.dumps(p0["figures"][0], ensure_ascii=False)[:300])


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