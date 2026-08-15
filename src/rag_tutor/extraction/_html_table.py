#!/usr/bin/env python3
"""Conversion d'une table HTML (<table>...</table>) -> table Markdown pipe.

Utilitaire partage entre normalizer.py (unification des anciens .md) et les
pipelines d'extraction. Les '|' des cellules (ex. |z| en LaTeX) sont echappes ;
le LaTeX des cellules est preserve tel quel.
"""

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
            self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._in = False

    def handle_data(self, data):
        if self._in:
            self._cell.append(data)


def html_table_to_md(html):
    """<table><tr><td>..</td></tr></table>  ->  table Markdown pipe.

    Les '|' des cellules (ex. |z| en LaTeX) sont echappes ; le LaTeX est garde
    tel quel. En cas d'echec de parsing, le HTML d'origine est retourne tel
    quel (mieux vaut du HTML brut qu'un crash)."""
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
