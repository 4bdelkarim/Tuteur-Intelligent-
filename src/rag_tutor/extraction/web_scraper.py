#!/usr/bin/env python3
"""
web_to_markdown.py — RAG ingestion: fetch web pages and convert them to
embedding-ready Markdown.

Design goal: a GENERIC, extensible core with ONE extractor per site. No
site-specific (e.g. Sphinx) assumptions leak into the common code; all d2l.ai
specifics are confined to ``D2lExtractor``.

Each commented section below is independently splittable into its own module:

  [core.fetch]         fetch_page()              static fetch (+ disabled JS hook)
  [core.model]         Document                  markdown + metadata, front-matter
  [core.noise]         filter_noise()            aggressive output-cell noise filter
  [core.serialize]     html_to_markdown()        shared markdownify serializer
  [core.registry]      register / get_extractor  {domain -> Extractor} dispatch
  [extractors.base]    BaseExtractor             contract only (stays generic)
  [extractors.generic] GenericExtractor          trafilatura fallback (unknown sites)
  [extractors.d2l]     D2lExtractor              ALL Sphinx/d2l specifics live here
  [mathml]             mathml_to_latex()         used only for rendered-DOM fallback
  [core.save]          save_document()

Adding a new site = write a `BaseExtractor` subclass, decorate it with
`@register`, set `DOMAINS`. Unknown domains fall back to `GenericExtractor`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import MarkdownConverter

DEFAULT_UA = "Mozilla/5.0 (compatible; RAG-ingest/1.0; +https://example.org/bot)"


# ===========================================================================
# [core.fetch]
# ===========================================================================
def fetch_page(url: str, *, render_js: bool = False, timeout: int = 20,
               session: "requests.Session | None" = None) -> str:
    """Return raw HTML for *url*.

    Static GET by default. This is REQUIRED for d2l.ai: a static fetch returns
    the raw LaTeX (``\\(...\\)``) *before* MathJax runs. Rendering the JS would
    replace it with non-recoverable CHTML. ``render_js`` is an opt-in hook for
    future JS-only sites; it is contra-indicated for d2l and disabled here.
    """
    if render_js:
        return _render_with_playwright(url, timeout=timeout)
    sess = session or requests
    resp = sess.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def _render_with_playwright(url: str, *, timeout: int = 20) -> str:
    """Optional JS-rendering fallback (lazy import). NOT used for d2l."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "render_js=True requires playwright "
            "(pip install playwright && playwright install chromium)"
        ) from e
    with sync_playwright() as p:  # pragma: no cover
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
        html = page.content()
        browser.close()
    return html


# ===========================================================================
# [core.model]
# ===========================================================================
@dataclass
class Document:
    markdown: str
    metadata: dict = field(default_factory=dict)

    def front_matter(self) -> str:
        # YAML is a superset of JSON, so json.dumps() yields valid, safely
        # quoted YAML scalars (handles quotes / unicode / None) with no PyYAML
        # dependency.
        lines = ["---"]
        for k, v in self.metadata.items():
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        lines.append("---")
        return "\n".join(lines)

    def render(self) -> str:
        return f"{self.front_matter()}\n\n{self.markdown.strip()}\n"


# ===========================================================================
# [core.noise]  — generic: applies to any site's code-output cells
# ===========================================================================
DEFAULT_NOISE_PATTERNS = [
    r"^\s*\[\d{2}:\d{2}:\d{2}\]",            # MXNet log timestamps
    r"Using Pooled .*StorageManager",
    r"No GPU/TPU found",
    r"falling back to CPU",
    r"TF_CPP_MIN_LOG_LEVEL",
    r"\bWARNING\b", r"UserWarning", r"DeprecationWarning", r"FutureWarning",
    r"Could not load dynamic library",
    r"(?i)\b(cuda|cudnn|cuinit|xla)\b",
    r"\d+%\|",                               # tqdm progress bars
    r"\bit/s\b",
]


def filter_noise(text: str, patterns: "list[str] | None" = None) -> str:
    """Drop noisy lines from an output block. Returns '' if nothing useful remains."""
    rx = [re.compile(p) for p in (patterns or DEFAULT_NOISE_PATTERNS)]
    kept = [ln for ln in text.split("\n") if not any(r.search(ln) for r in rx)]
    return "\n".join(kept).strip("\n")


# ===========================================================================
# [core.serialize]  — generic HTML subtree -> Markdown
# ===========================================================================
_CHROME_ALTS = {"", "copy to clipboard", "dive into deep learning"}


class _Serializer(MarkdownConverter):
    """markdownify with sane defaults; drops chrome images by alt text."""

    def convert_img(self, el, text, *args, **kwargs):  # version-proof signature
        alt = (el.attrs.get("alt") or "").strip()
        if alt.lower() in _CHROME_ALTS:
            return ""
        src = el.attrs.get("src") or ""
        return f"![{alt}]({src})"


_SERIALIZER = _Serializer(heading_style="ATX", bullets="-")


def html_to_markdown(node) -> str:
    """Headings, paragraphs, lists, tables, inline formatting, links -> Markdown."""
    return _SERIALIZER.convert(str(node))


# ===========================================================================
# [core.registry]
# ===========================================================================
_EXTRACTORS: "list[type[BaseExtractor]]" = []


def register(cls: "type[BaseExtractor]") -> "type[BaseExtractor]":
    _EXTRACTORS.append(cls)
    return cls


def get_extractor(url: str) -> "BaseExtractor":
    host = (urlparse(url).netloc or "").lower()
    host = host[4:] if host.startswith("www.") else host
    for cls in _EXTRACTORS:
        for dom in cls.DOMAINS:
            if host == dom or host.endswith("." + dom):
                return cls()
    return GenericExtractor()


# ===========================================================================
# [extractors.base]  — contract only; stays domain-agnostic
# ===========================================================================
class BaseExtractor(ABC):
    DOMAINS: "tuple[str, ...]" = ()

    @abstractmethod
    def extract(self, html: str, url: str) -> Document:
        ...


# ===========================================================================
# [extractors.generic]  — trafilatura fallback for unknown (non-Sphinx) sites
# ===========================================================================
def _protect_images(html: str):
    """trafilatura perd le contenu des <img> imbriquees dans des tableaux HTML
    bruts (ex. <center><img src=".."/></center> dans une cellule) -> on les
    remplace par un token TEXTE avant extraction (meme technique que les
    placeholders MATH/CODE de D2lExtractor), et on restaure ![alt](src) apres.

    IMPORTANT : <center> est une balise presentationnelle que trafilatura
    traite comme du "chrome" et supprime (avec son contenu) en contexte de
    page complete -> on la DEBALLE (unwrap) avant de poser les tokens, sinon
    meme le token disparait avec elle."""
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all("center"):
        c.unwrap()
    ph = {}
    for i, img in enumerate(soup.find_all("img")):
        src = img.get("data-src") or img.get("src") or ""   # data-src : lazy-load
        if not src:
            continue
        src = re.sub(r"\{\{\s*site\.baseurl\s*\}\}", "", src)  # variable Jekyll non resolue
        alt = (img.get("alt") or "").strip()
        token = f"ZZIMGZZ{i}ZZ"
        ph[token] = f"![{alt}]({src})"
        img.replace_with(NavigableString(f" {token} "))
    return str(soup), ph


def _restore_images(md: str, ph: dict) -> str:
    for token, value in ph.items():
        md = md.replace(token, value)
    return md


class GenericExtractor(BaseExtractor):
    DOMAINS = ()

    def extract(self, html: str, url: str) -> Document:
        try:
            import trafilatura
        except ImportError as e:
            raise RuntimeError(
                "GenericExtractor needs trafilatura (pip install trafilatura)"
            ) from e
        html, ph = _protect_images(html)
        md = trafilatura.extract(
            html,
            output_format="markdown",
            include_tables=True,
            include_links=True,
            include_images=True,
            favor_recall=True,
        ) or ""
        md = _restore_images(md, ph)
        meta = {"source_url": url, "title": None, "date": None, "extractor": "generic"}
        try:
            tmeta = trafilatura.extract_metadata(html)
            if tmeta:
                meta["title"] = tmeta.title
                meta["date"] = tmeta.date
        except Exception:
            pass
        return Document(markdown=md.strip(), metadata=meta)


# ===========================================================================
# [extractors.d2l]  — ALL Sphinx/d2l-specific logic confined here
# ===========================================================================
@register
class D2lExtractor(BaseExtractor):
    DOMAINS = ("d2l.ai",)

    MAIN_SELECTORS = ["div[role=main]", "main", "div.document"]
    DROP_SELECTORS = [
        "a.headerlink",            # the ¶ permalinks
        "div.d2l-tabs",            # Colab launcher tabs (inside the <h1>)
        "div.mdl-tooltip",         # MDL tooltips ("Open the notebook in …")
        "a.copybtn", ".copybtn",   # sphinx-copybutton
        "script", "style",
    ]
    # <a> whose href matches a notebook launcher -> drop the whole link (+ its
    # button). The SageMaker launcher is a *bare* <a> sibling of div.d2l-tabs in
    # the <h1>, so a class/selector alone misses it.
    DROP_LINK_PATTERNS = [
        r"colab\.research\.google",
        r"studiolab\.sagemaker",
        r"sagemaker\.aws",
    ]
    # links whose enclosing <p> is pure boilerplate -> drop the whole paragraph
    DROP_PARAGRAPH_LINK_PATTERNS = [r"discuss\.d2l\.ai"]   # per-page "Discussions" forum link

    # ---- template method ----
    def extract(self, html: str, url: str) -> Document:
        soup = BeautifulSoup(html, "lxml")
        meta = self._metadata(soup, url)
        main = self._select_main(soup)
        if main is None:
            raise ValueError("d2l: main content container not found")
        self._collapse_tabs(main)        # keep only the active (PyTorch) panel
        self._drop_chrome(main)
        ph: "dict[str, str]" = {}
        self._handle_math(main, ph)
        self._handle_code(main, ph)
        md = html_to_markdown(main)
        md = self._restore(md, ph)
        return Document(markdown=_tidy(md), metadata=meta)

    # ---- hooks ----
    def _select_main(self, soup):
        for sel in self.MAIN_SELECTORS:
            el = soup.select_one(sel)
            if el is not None:
                return el
        return None

    def _collapse_tabs(self, main):
        """Framework tabs: <div class="mdl-tabs"> with one .mdl-tabs__panel.is-active.
        Keep only PyTorch (the active panel); drop the others + the tab bar."""
        for tabs in main.select("div.mdl-tabs"):
            bar = tabs.select_one(".mdl-tabs__tab-bar")
            if bar is not None:
                bar.decompose()
            for panel in tabs.select(".mdl-tabs__panel"):
                if "is-active" not in (panel.get("class") or []):
                    panel.decompose()

    def _drop_chrome(self, main):
        for sel in self.DROP_SELECTORS:
            for el in main.select(sel):
                el.decompose()
        # notebook launcher links (Colab / SageMaker) — remove the whole <a> + button
        for pat in self.DROP_LINK_PATTERNS:
            for a in main.find_all("a", href=re.compile(pat)):
                a.decompose()
        # boilerplate paragraphs that exist only to hold an external link (Discussions)
        for pat in self.DROP_PARAGRAPH_LINK_PATTERNS:
            for a in main.find_all("a", href=re.compile(pat)):
                (a.find_parent("p") or a).decompose()
        # in-text internal cross-refs -> keep visible text, drop the (dead-on-chunk) link
        for a in main.select("a.reference.internal"):
            a.replace_with(NavigableString(a.get_text()))

    def _handle_math(self, main, ph):
        for selector, inline in (("span.math", True), ("div.math", False)):
            for el in main.select(selector):
                latex = self._extract_latex(el)
                token = _token("MATH", len(ph))
                ph[token] = f"${latex}$" if inline else f"\n$$\n{latex}\n$$\n"
                if inline:
                    el.replace_with(NavigableString(token))
                else:
                    el.replace_with(_block_placeholder(token))

    def _handle_code(self, main, ph):
        # inputs: highlight-<lang> but NOT .output
        for el in main.select('div[class*="highlight-"]:not(.output)'):
            lang = self._lang(el)
            code = _block_text(el.select_one("pre"))
            token = _token("CODE", len(ph))
            ph[token] = (f"\n```{lang}\n{code}\n```\n") if code else ""
            el.replace_with(_block_placeholder(token))
        # outputs: .output  (noise-filtered, dropped if empty)
        # outputs: .output (MODIFIÉ: On ignore l'output en mettant une chaîne vide)
        for el in main.select("div.output"):
            token = _token("CODE", len(ph))
            ph[token] = ""  # Suppression de l'output
            el.replace_with(_block_placeholder(token))

    # ---- helpers ----
    @staticmethod
    def _lang(el) -> str:
        for c in el.get("class", []):
            if c.startswith("highlight-"):
                lang = c[len("highlight-"):]
                return "" if lang == "default" else lang
        return ""

    def _extract_latex(self, el) -> str:
        """Dual-mode: raw \\(...\\) (production / requests.get) OR MathML (rendered DOM)."""
        # rendered / saved-page form -> assistive MathML -> LaTeX
        if el.select_one("mjx-container") is not None:
            mml = el.select_one("mjx-assistive-mml math") or el.select_one("math")
            if mml is not None:
                return mathml_to_latex(mml).strip()
        # static form -> strip the \( \) / \[ \] delimiters
        raw = el.get_text()
        raw = re.sub(r"^\s*\\[(\[]\s*", "", raw)
        raw = re.sub(r"\s*\\[)\]]\s*$", "", raw)
        return raw.strip()

    def _metadata(self, soup, url) -> dict:
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.split("—")[0].strip()
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True).replace("¶", "") if h1 else None
        chapter = section_number = None
        if title:
            m = re.match(r"(\d+(?:\.\d+)*)", title)
            if m:
                section_number = m.group(1)
                chapter = section_number.split(".")[0]
        return {
            "source_url": url,
            "title": title,
            "chapter": chapter,
            "section_number": section_number,
            "date": None,
            "extractor": "d2l",
        }

    @staticmethod
    def _restore(md: str, ph: "dict[str, str]") -> str:
        for token, value in ph.items():
            md = md.replace(token, value)
        return md


# ===========================================================================
# [mathml]  MathML -> LaTeX  (compact; d2l's simple presentation-MathML subset)
# Used ONLY for the rendered/saved-HTML fallback. Production uses raw LaTeX.
# ===========================================================================
_MO_MAP = {
    "\u2212": "-", "\u00d7": r"\times", "\u22c5": r"\cdot", "\u2208": r"\in",
    "\u2209": r"\notin", "\u2264": r"\leq", "\u2265": r"\geq", "\u2260": r"\neq",
    "\u2061": "", "\u2062": "", "\u2192": r"\to", "\u21a6": r"\mapsto",
    "\u2211": r"\sum", "\u220f": r"\prod", "\u222b": r"\int",
    "\u2202": r"\partial", "\u2207": r"\nabla", "\u221e": r"\infty",
}
_MI_MAP = {"\u211d": r"\mathbb{R}", "\u2115": r"\mathbb{N}", "\u2124": r"\mathbb{Z}",
           "\u211a": r"\mathbb{Q}", "\u2102": r"\mathbb{C}"}


def mathml_to_latex(node) -> str:
    if isinstance(node, NavigableString):
        s = str(node)
        return s if s.strip() else ""
    name = (node.name or "").lower()
    if name == "annotation":
        return ""
    elems = [c for c in node.children if isinstance(c, Tag)]
    children_latex = lambda: "".join(mathml_to_latex(c) for c in node.children)

    if name in ("math", "mrow", "mstyle", "semantics", "mpadded", "mphantom"):
        return children_latex()
    if name == "mi":
        t = node.get_text()
        return _MI_MAP.get(t, t)
    if name == "mn":
        return node.get_text()
    if name == "mo":
        t = node.get_text().strip()
        return _MO_MAP.get(t, t)
    if name == "mtext":
        return r"\text{%s}" % node.get_text()
    if name == "msup" and len(elems) >= 2:
        return "%s^{%s}" % (mathml_to_latex(elems[0]), mathml_to_latex(elems[1]))
    if name == "msub" and len(elems) >= 2:
        return "%s_{%s}" % (mathml_to_latex(elems[0]), mathml_to_latex(elems[1]))
    if name == "msubsup" and len(elems) >= 3:
        return "%s_{%s}^{%s}" % tuple(mathml_to_latex(e) for e in elems[:3])
    if name == "mfrac" and len(elems) >= 2:
        return r"\frac{%s}{%s}" % (mathml_to_latex(elems[0]), mathml_to_latex(elems[1]))
    if name == "msqrt":
        return r"\sqrt{%s}" % children_latex()
    if name == "mroot" and len(elems) >= 2:
        return r"\sqrt[%s]{%s}" % (mathml_to_latex(elems[1]), mathml_to_latex(elems[0]))
    if name == "mfenced":
        return r"\left(%s\right)" % children_latex()
    # fallback: recurse / leaf text
    return children_latex() if elems else node.get_text()


# ===========================================================================
# small shared helpers
# ===========================================================================
def _token(kind: str, i: int) -> str:
    # alnum-only + ZZ sentinels: markdownify never escapes it, no prefix collisions
    return f"ZZ{kind}{i}ZZ"


def _block_placeholder(token: str) -> Tag:
    """A standalone block <p> placeholder so the serializer keeps it on its own line."""
    return BeautifulSoup(f"<p>{token}</p>", "html.parser").p


def _block_text(pre) -> str:
    """Plain text of a <pre>, stripping Pygments spans but preserving indentation."""
    if pre is None:
        return ""
    lines = pre.get_text().split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _tidy(md: str) -> str:
    md = re.sub(r"[ \t]+\n", "\n", md)      # trailing whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)      # collapse blank runs
    md = md.replace("¶", "")
    return md.strip() + "\n"


# ===========================================================================
# [core.save]
# ===========================================================================
def _slug(url: str, doc: Document) -> str:
    path = urlparse(url).path
    stem = Path(path).stem if path else ""
    if stem and stem != "index":
        parent = Path(path).parent.name
        return f"{parent}_{stem}" if parent and parent not in ("", "/") else stem
    title = doc.metadata.get("title") or "document"
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "document"


def save_document(doc: Document, out_dir: str, url: "str | None" = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = url or doc.metadata.get("source_url") or "document"
    path = out / f"{_slug(url, doc)}.md"
    path.write_text(doc.render(), encoding="utf-8")
    return path


# ===========================================================================
# orchestration + CLI
# ===========================================================================
def process(*, url: "str | None" = None, file: "str | None" = None,
            canonical_url: "str | None" = None, render_js: bool = False) -> Document:
    if file:
        html = Path(file).read_text(encoding="utf-8", errors="replace")
        ref = canonical_url or url or f"file://{file}"
    elif url:
        html = fetch_page(url, render_js=render_js)
        ref = canonical_url or url
    else:
        raise ValueError("Provide url= or file=")
    return get_extractor(ref).extract(html, ref)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fetch/convert a web page to embedding-ready Markdown.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--file", help="local HTML file (for validation)")
    ap.add_argument("--canonical-url",
                    help="source URL to record / dispatch on when using --file")
    ap.add_argument("--out", default="output")
    ap.add_argument("--render-js", action="store_true")
    ap.add_argument("--print", action="store_true", help="print markdown to stdout")
    a = ap.parse_args(argv)
    doc = process(url=a.url, file=a.file,
                  canonical_url=a.canonical_url, render_js=a.render_js)
    path = save_document(doc, a.out, url=a.canonical_url or a.url)
    print(f"[ok] extractor={doc.metadata.get('extractor')} -> {path}", file=sys.stderr)
    if a.print:
        print(doc.render())
    return doc


if __name__ == "__main__":
    main()