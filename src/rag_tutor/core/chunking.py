#!/usr/bin/env python3
"""
chunking.py — Chunking PARENT-CHILD sur le format UNIFIE (sortie normalizer.py).

SEULE RESPONSABILITE DE CE MODULE : decouper un corpus deja normalise en chunks
parent/enfant, enrichis des metadata necessaires, prets a etre embeddes.
AUCUN embedding, AUCUNE ecriture en base vectorielle, AUCUN retrieval ici -- ces
taches vivent dans d'autres modules (embeddings.py, vector_store.py/ingest.py,
retriever.py) qui consomment la sortie de ce module sans la recalculer.

Prerequis : les fichiers ont deja ete unifies par normalizer.py. Ce script ne fait
AUCUNE detection de format -> il suppose UNE seule convention partout :
  - front-matter YAML toujours present (source_type: pdf|web)
  - pages   : <!-- loc page=N -->                    (absent si web)
  - figures : --- [FIGURE] --- ... --- [/FIGURE] ---   (absent si web)
  - codes   : ``` ... ```                              (blocs de code markdown)
  - titres  : '#'..'######' markdown
  - tables  : markdown pipe '| ... |'

Principe (inchange) : on CHERCHE petit, on GENERE grand.
  - ENFANTS  : petits chunks (~400 car.) indexes et embeddes -> recuperation PRECISE.
  - PARENTS  : la SECTION complete a laquelle appartient l'enfant -> contexte COMPLET.
  Formules $$...$$, blocs [FIGURE], blocs ```...``` et tables markdown restent
  ATOMIQUES (jamais coupes).

  pip install pyyaml
  python normalizer.py ./bruts/ --out ./processed/          # etape 1 (separee)
  python chunking.py ./processed/                 # etape 2 (ce script) -- stats seulement

API publique :
  parse_file(md_path)   -> (parents: dict, children: list[dict])   # UN fichier
  chunk_corpus(path)    -> (parents: dict, children: list[dict])   # dossier ou fichier

Chaque enfant contient deja "embed_text" (texte final a passer a l'embedder) et
toutes les metadata d'indexation -- rien d'autre a calculer en aval.

Sortie (objets Python en memoire -- ce module n'ecrit rien sur disque) :
  parents  : dict[parent_id -> {text, source, source_type, section, page_start, page_end, source_url}]
  children : list[{id, text, embed_text, parent_id, source, source_type, page, section, child_type}]
"""

import re
import statistics
import argparse
from pathlib import Path

# =====================================================
# CONFIG (chunking uniquement -- DB_DIR / COLLECTION_NAME / EMBEDDING_MODEL
# et tout ce qui concerne l'embedding/la base vectorielle n'ont plus leur place ici)
# =====================================================

CHILD_TARGET  = 400
CHILD_MAX     = 750
CHILD_OVERLAP = 80    # caracteres de chevauchement entre enfants consecutifs (embed_text
                        # seulement -- le text brut reste sans overlap pour ne pas dupliquer
                        # le contenu dans l'assemblage parent). Ameliore le rappel quand une
                        # information est a cheval sur deux enfants.

# --- Regex : UNE seule convention (fichiers deja unifies par normalizer.py) ---
FRONTMATTER_RE  = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)
PAGE_RE         = re.compile(r'(?im)^<!--\s*loc\s+page=(\d+)\s*-->\s*$')
FORMULA_RE      = re.compile(r'\$\$.*?\$\$', re.DOTALL)
FIG_RE          = re.compile(r'---\s*\[FIGURE\]\s*---.*?---\s*\[/FIGURE\]\s*---', re.DOTALL | re.IGNORECASE)
CODE_BLOCK_RE   = re.compile(r'```.*?```', re.DOTALL)
TABLE_RE        = re.compile(r'(?:^\|.*\|[ \t]*\n){2,}', re.MULTILINE)
HEADER_RE       = re.compile(r'^#{1,6}\s+\S')


# =====================================================
# 1) FRONT-MATTER (toujours present, ecrit par normalizer.py)
# =====================================================

def split_frontmatter(text):
    """Extrait le front-matter YAML. Garantit que source_type est toujours present
    et correct, avec fallback explicite si absent du YAML (ne devrait jamais arriver
    apres normalizer.py, mais filet de securite)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        import yaml
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}

    # P3: s'assurer que source_type est toujours present et valide.
    # normalizer.py ecrit TOUJOURS source_type dans le front-matter -> si absent,
    # c'est un fichier non normalise ; on tente une detection heuristique.
    if meta.get("source_type") not in ("pdf", "web"):
        # Heuristique : presence de marqueurs de page -> pdf, sinon web
        meta["source_type"] = "pdf" if re.search(
            r'(?im)^<!--\s*(?:loc\s+)?page[=\s]', text
        ) else "web"

    return meta, text[m.end():]


# =====================================================
# 2) TOKENISATION : page / formule / figure / code / table / prose
#    (ordre preserve, PUR -- jamais de chevauchement entre unites atomiques)
# =====================================================

def tokenize(text):
    """Decoupe le texte en segments preserves :
    - page      : marqueur <!-- loc page=N -->
    - formula   : $$...$$  (LaTeX, jamais coupe)
    - figure    : --- [FIGURE] --- ... --- [/FIGURE] ---
    - code_block: ```...```  (P1: blocs de code markdown, jamais coupes)
    - table     : table markdown pipe (2+ lignes consecutives |...|)
    - prose     : tout le reste
    """
    events = []
    for m in PAGE_RE.finditer(text):
        events.append((m.start(), m.end(), "page", int(m.group(1))))
    for m in FORMULA_RE.finditer(text):
        events.append((m.start(), m.end(), "formula", m.group(0)))
    for m in FIG_RE.finditer(text):
        events.append((m.start(), m.end(), "figure", m.group(0)))
    for m in CODE_BLOCK_RE.finditer(text):
        events.append((m.start(), m.end(), "code_block", m.group(0)))
    for m in TABLE_RE.finditer(text):
        events.append((m.start(), m.end(), "table", m.group(0)))
    events.sort()

    segs, last = [], 0
    for s, e, kind, payload in events:
        if s < last:
            continue  # skip si chevauchement (un token dans un autre -- ne devrait pas arriver)
        if s > last:
            segs.append(("prose", text[last:s], None))
        segs.append((kind, payload, None))
        last = e
    if last < len(text):
        segs.append(("prose", text[last:], None))
    return segs


def is_header(line):
    """P2: Detecte un titre markdown. La protection contre les faux headers
    (commentaires de code '# ceci est un commentaire') est assuree en AMONT par
    tokenize() : les blocs de code ```...``` sont extraits comme unites atomiques
    'code_block' et n'arrivent JAMAIS dans la prose. Le simple match du pattern
    markdown est donc suffisant et sans risque de faux positif."""
    return bool(HEADER_RE.match(line))


# =====================================================
# 3) CONSTRUCTION DES PARENTS (= sections)
# =====================================================

def build_parents(text):
    parents = []
    cur = None
    current_page = None
    current_section = "preambule"

    def start_parent(section, page):
        nonlocal cur
        cur = {"section": section, "page_start": page, "page_end": page, "segments": []}
        parents.append(cur)

    def add(typ, payload, page):
        nonlocal cur
        if cur is None:
            start_parent(current_section, page)
        cur["segments"].append((typ, payload, page))
        if page is not None:
            cur["page_end"] = page
            if cur["page_start"] is None:
                cur["page_start"] = page

    for kind, payload, _ in tokenize(text):
        if kind == "page":
            current_page = payload
            if cur is not None:
                cur["page_end"] = current_page
                if cur["page_start"] is None:
                    cur["page_start"] = current_page
            continue

        # P1: code_block, formula, figure, table -> atomiques, ajoutes tels quels
        if kind in ("formula", "figure", "table", "code_block"):
            add(kind, payload, current_page)
            continue

        # prose : on cherche les titres ligne par ligne
        buf = []
        for line in payload.splitlines(keepends=True):
            if is_header(line):
                if buf:
                    add("prose", "".join(buf), current_page); buf = []
                current_section = line.strip().lstrip("#").strip()
                start_parent(current_section, current_page)
                buf = [line]
            else:
                buf.append(line)
        if buf:
            add("prose", "".join(buf), current_page)

    return [p for p in parents if p["segments"]]


# =====================================================
# 4) DECOUPAGE EN ENFANTS (blocs atomiques preserves)
# =====================================================

def prose_units(text, hardmax):
    """Decoupe la prose en unites (paragraphes puis phrases). Les blocs de code
    ne passent plus par ici (P1: ils sont deja extraits comme code_block dans tokenize)."""
    units = []
    for para in re.split(r'\n\s*\n', text):
        if not para.strip():
            continue
        if len(para) <= hardmax:
            units.append(para.strip() + "\n\n")
        else:
            cur = ""
            for s in re.split(r'(?<=[\.\!\?])\s+', para):
                if len(cur) + len(s) <= hardmax:
                    cur = (cur + " " + s).strip()
                else:
                    if cur:
                        units.append(cur + " ")
                    cur = s
            if cur:
                units.append(cur + "\n\n")
    return units


def build_children(segments, target, hardmax, overlap=CHILD_OVERLAP):
    """Construit les enfants a partir des segments d'une section.
    overlap: nombre de caracteres du child precedent a inclure au debut de
    l'embed_text du child suivant (P4). Le champ 'text' reste sans overlap
    pour ne pas dupliquer le contenu dans l'assemblage parent."""
    units = []
    for typ, text, page in segments:
        if typ in ("formula", "figure", "table", "code_block"):
            units.append((text, page, typ))
        else:
            for u in prose_units(text, hardmax):
                units.append((u, page, "prose"))

    children, buf, size, pg, types = [], [], 0, None, set()

    def flush():
        nonlocal buf, size, pg, types
        if buf:
            children.append({
                "text": "".join(buf).strip(),
                "page": pg,
                "child_type": "+".join(sorted(types)) if types else "prose",
            })
        buf, size, pg, types = [], 0, None, set()

    for u, page, typ in units:
        if pg is None:
            pg = page
        if buf and size + len(u) > target:
            flush(); pg = page
        buf.append(u); size += len(u); types.add(typ)
        if typ in ("formula", "figure", "table", "code_block") and size >= target:
            flush()
        elif size >= hardmax:
            flush()
    flush()

    # P4: ajout d'overlap dans embed_text (pas dans text, qui reste propre pour
    # l'assemblage parent). On pre-calcule les suffixes d'overlap.
    if overlap > 0 and len(children) > 1:
        for i in range(1, len(children)):
            prev_text = children[i - 1]["text"]
            if len(prev_text) > overlap:
                children[i]["_overlap_prefix"] = prev_text[-overlap:] + "\n"
            else:
                children[i]["_overlap_prefix"] = prev_text + "\n"

    return children


# =====================================================
# 5) CHUNKING + METADATA -- API PUBLIQUE (aucun embedding, aucun I/O base)
# =====================================================

def parse_file(md_path, child_target=CHILD_TARGET, child_max=CHILD_MAX,
               child_overlap=CHILD_OVERLAP):
    """Pur (sans embeddings) : renvoie (parents_store, children) pour UN fichier deja unifie."""
    source = md_path.stem
    raw = md_path.read_text(encoding="utf-8")
    meta, text = split_frontmatter(raw)
    source_type = meta.get("source_type", "pdf")

    raw_parents = build_parents(text)

    parents_store, children = {}, []
    for idx, p in enumerate(raw_parents):
        pid = f"{source}__sec{idx:03d}"
        parent_text = "".join(seg[1] for seg in p["segments"]).strip()
        if not parent_text:
            continue
        parents_store[pid] = {
            "text": parent_text,
            "source": source,
            "source_type": source_type,
            "section": p["section"],
            "page_start": p["page_start"],
            "page_end": p["page_end"],
            "source_url": meta.get("source_url"),
            "title": meta.get("title"),          # titre de la page (web) pour les citations
            "source_id": meta.get("source_id"),  # nom du fichier PDF (ex: 02_NN.pdf)
        }
        for j, ch in enumerate(build_children(p["segments"], child_target, child_max,
                                                overlap=child_overlap)):
            cid = f"{pid}__c{j:03d}"
            page_tag = f"page {ch['page']}" if ch["page"] is not None else "page ?"
            header_prefix = f"[{source} | {p['section']} | {page_tag}]\n"

            # P4: embed_text recoit le prefix d'overlap du child precedent s'il existe
            overlap_prefix = ch.pop("_overlap_prefix", "")
            embed_text = header_prefix + overlap_prefix + ch["text"]

            children.append({
                "id": cid,
                "text": ch["text"],
                "embed_text": embed_text,
                "parent_id": pid,
                "source": source,
                "source_type": source_type,
                "page": ch["page"],
                "section": p["section"],
                "child_type": ch["child_type"],
            })
    return parents_store, children


def chunk_corpus(path, child_target=CHILD_TARGET, child_max=CHILD_MAX,
                 child_overlap=CHILD_OVERLAP):
    """Pur : meme contrat que parse_file, mais sur un dossier entier (ou un fichier unique).
    C'est CETTE fonction que ingest.py et les tests doivent appeler -- jamais parse_file
    directement en boucle ailleurs, sinon la logique d'agregation se duplique."""
    files = list(path.rglob("*.md")) if path.is_dir() else [path]
    all_parents, all_children = {}, []
    for f in files:
        ps, cs = parse_file(f, child_target, child_max, child_overlap)
        all_parents.update(ps)
        all_children.extend(cs)
    return all_parents, all_children


# =====================================================
# STATS (diagnostic pur -- prend des chunks deja construits, ne touche a rien d'autre)
# =====================================================

def _dist(label, values):
    if not values:
        print(f"  {label:<26} (aucun)"); return
    print(f"  {label:<26} min={min(values):>4}  med={int(statistics.median(values)):>4}  "
          f"moy={int(statistics.mean(values)):>4}  max={max(values):>5}")


def print_stats(parents, children):
    print("=" * 60)
    print("STATISTIQUES DE DECOUPAGE PARENT-CHILD")
    print("=" * 60)
    print(f"  Parents (sections) : {len(parents)}")
    print(f"  Enfants (chunks)   : {len(children)}")
    if parents:
        from collections import Counter
        src_types = Counter(p["source_type"] for p in parents.values())
        print(f"  Repartition source_type (parents) : {dict(src_types)}")
        per = {}
        for c in children:
            per[c["parent_id"]] = per.get(c["parent_id"], 0) + 1
        _dist("Tailles des parents (car.)", [len(p["text"]) for p in parents.values()])
        _dist("Tailles des enfants (car.)", [len(c["text"]) for c in children])
        _dist("Enfants par parent", list(per.values()))
        types = Counter(c["child_type"] for c in children)
        print(f"  Types d'enfants            : {dict(types)}")
    print("=" * 60)


# =====================================================
# MAIN -- diagnostic seul (chunking + stats). Pas d'embedding, pas d'indexation ici :
# c'est le role de ingest.py, qui importera chunk_corpus() depuis ce module.
# =====================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Chunking parent-child sur corpus deja unifie (normalizer.py). "
                     "N'embedde rien, n'ecrit rien en base : affiche les stats de decoupage."
    )
    ap.add_argument("path", help="dossier (ou fichier) de sortie de normalizer.py")
    ap.add_argument("--child-target", type=int, default=CHILD_TARGET)
    ap.add_argument("--child-max", type=int, default=CHILD_MAX)
    ap.add_argument("--child-overlap", type=int, default=CHILD_OVERLAP,
                    help=f"chevauchement entre enfants consecutifs en embed_text (defaut {CHILD_OVERLAP})")
    args = ap.parse_args()

    parents, children = chunk_corpus(Path(args.path), args.child_target, args.child_max,
                                      args.child_overlap)
    print_stats(parents, children)
