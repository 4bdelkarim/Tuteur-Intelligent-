#!/usr/bin/env python3
"""
web_crawler.py — crawler de site, pilote web_scraper.py (ton scraper).

Séparation des rôles (c'est tout l'intérêt d'avoir deux fichiers) :

    CE FICHIER (web_crawler.py)     -> trouve TOUTES les pages du site.
                                       Part de l'URL racine, suit les liens
                                       internes en largeur (BFS), déduplique.
    web_scraper.py (ton scraper)-> pour chaque page trouvée : extrait le
                                       contenu et l'enregistre en Markdown.
                                       Importé tel quel, JAMAIS modifié.

Le crawler ne connaît rien du HTML des pages : il se contente de découvrir des
URLs et de les passer à ton scraper via ses fonctions publiques
(`fetch_page`, `get_extractor`, `save_document`). Ajouter un site = ajouter un
extracteur dans ton scraper, rien à changer ici.

Exemples :
    # tout le cours NYU-DLSP21 (le préfixe /NYU-DLSP21 est détecté tout seul)
    python web_crawler.py https://atcold.github.io/NYU-DLSP21/ --out corpus

    # tout le livre d2l
    python web_crawler.py https://d2l.ai/ --out corpus --max-pages 600

    # plusieurs sites en une passe, chacun reste dans son périmètre
    python web_crawler.py https://d2l.ai/ https://atcold.github.io/NYU-DLSP21/ --out corpus
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# --- ton scraper, importé et réutilisé tel quel -----------------------------
# (le fichier s'appelle web_scraper.py et est dans le même dossier,
#  ou accessible via le PYTHONPATH)
try:
    from . import web_scraper as scraper
except ImportError:  # exécuté comme script autonome (python web_crawler.py)
    import web_scraper as scraper

DEFAULT_UA = getattr(scraper, "DEFAULT_UA",
                     "Mozilla/5.0 (compatible; RAG-ingest/1.0)")

# Extensions qui ne sont pas des pages HTML de contenu -> jamais suivies.
SKIP_EXTENSIONS = {
    ".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".webp", ".ipynb", ".txt", ".css", ".js", ".json", ".xml",
    ".mp4", ".webm", ".mp3", ".woff", ".woff2", ".ttf", ".eot", ".rst",
}

# Motif d'URL qui reconnaît une PAGE DE CONTENU DE COURS, par site. Seuls les
# liens dont le CHEMIN correspond sont suivis ET enregistrés ; le reste (accueil,
# références, FAQ, liens externes...) est ignoré. Vérifié sur les URLs réelles :
#   - d2l   : les chapitres/sections vivent tous sous /chapter_xxx/....html
#             (ex. /chapter_preliminaries/ndarray.html) -> motif "/chapter_"
#   - NYU   : les semaines vivent sous /weekNN/... (ex. /en/week02/02-3/)
#             -> motif "/week\d"
# Un site absent de ce dict (et sans --include) => aucun filtre : on suit tout
# ce qui est dans le périmètre (comportement précédent), avec un avertissement.
CONTENT_PATTERNS = {
    "d2l.ai": r"/chapter_",
    "atcold.github.io": r"/week\d",
}


def content_pattern_for(host: str, user_include: "str | None") -> "re.Pattern | None":
    """--include (fourni par l'utilisateur) a priorité ; sinon le motif connu
    du site ; sinon None (pas de filtre de contenu)."""
    if user_include:
        return re.compile(user_include)
    pat = CONTENT_PATTERNS.get(host)
    return re.compile(pat) if pat else None


def is_content(url: str, content_re: "re.Pattern | None") -> bool:
    """Cette URL est-elle une page de contenu de cours ? Si aucun motif n'est
    défini pour le site, on considère tout comme du contenu (pas de filtre)."""
    if content_re is None:
        return True
    return bool(content_re.search(urlparse(url).path))


# ===========================================================================
# petits utilitaires
# ===========================================================================
def norm_host(netloc: str) -> str:
    """Normalise un netloc : minuscules, sans userinfo, port ni ``www.``."""
    host = (netloc or "").lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def norm_url(url: str) -> str:
    """Enlève le fragment (#section) : c'est la même page, pas une nouvelle
    URL à récupérer. Indispensable pour que la déduplication BFS marche."""
    stripped, _frag = urldefrag(url)
    return stripped.rstrip() if stripped else stripped


def default_path_prefix(seed_url: str) -> str:
    """Périmètre du crawl, déduit du seed.

    Sur les *.github.io (GitHub Pages), un même compte héberge plusieurs
    projets sous /projet1/, /projet2/... -> on se limite au premier segment
    du chemin (ex. /NYU-DLSP21). Sur un domaine dédié (d2l.ai) -> tout le
    domaine. Toujours surchargeable via --path-prefix."""
    p = urlparse(seed_url)
    host = norm_host(p.netloc)
    if host.endswith("github.io"):
        segs = [s for s in p.path.split("/") if s]
        if segs:
            return "/" + segs[0]
    return ""


def discover_links(soup: BeautifulSoup, base_url: str) -> "list[str]":
    """Tous les <a href> de la page, résolus en URL absolue.

    On scanne la page ENTIÈRE (pas seulement la zone de contenu) : sur un
    livre/cours, la table des matières / sidebar qui liste toutes les pages
    est presque toujours HORS du conteneur principal — c'est souvent le seul
    endroit où le plan complet du site est présent. (Vrai pour d2l comme pour
    NYU-DLSP21.)"""
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        out.append(urljoin(base_url, href))
    return out


def in_scope(url: str, root_host: str, path_prefix: str) -> bool:
    """L'URL est-elle crawlable : http(s), même hôte, dans le préfixe, non binaire."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if norm_host(p.netloc) != root_host:          # même site uniquement
        return False
    if path_prefix and not p.path.startswith(path_prefix):
        return False
    if any(p.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return True


class RobotsCache:
    """Un RobotFileParser par host, chargé à la première utilisation. Un
    robots.txt injoignable/absent = tout autorisé (lecture conventionnelle
    d'un robots.txt manquant : on ne bloque pas un site parce que son
    robots.txt ne répond pas)."""

    def __init__(self, session: requests.Session):
        self._session = session
        self._parsers: dict = {}

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        rp = self._parsers.get(root)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self._session.get(f"{root}/robots.txt",
                                      headers={"User-Agent": DEFAULT_UA}, timeout=10)
                rp.parse(r.text.splitlines() if r.status_code < 400 else [])
            except requests.RequestException:
                rp.parse([])
            self._parsers[root] = rp
        return rp.can_fetch(DEFAULT_UA, url)


# ===========================================================================
# le crawl
# ===========================================================================
def crawl(seeds: "list[str]", out_dir: str, *, max_pages: int = 500,
          delay: float = 0.5, respect_robots: bool = True,
          path_prefix: "str | None" = None,
          include: "str | None" = None) -> "list[Path]":
    """Parcours en largeur depuis un ou plusieurs seeds. Seules les pages de
    CONTENU DE COURS (voir CONTENT_PATTERNS / --include) sont suivies et
    enregistrées par ton scraper. Retourne la liste des fichiers écrits."""
    sess = requests.Session()
    robots = RobotsCache(sess) if respect_robots else None

    # Par host : périmètre de chemin + motif de contenu.
    scope: dict = {}
    content_re_of: dict = {}
    seen: set = set()
    queue: deque = deque()
    for s in seeds:
        u = norm_url(s)
        host = norm_host(urlparse(u).netloc)
        scope[host] = path_prefix if path_prefix is not None else default_path_prefix(u)
        content_re_of[host] = content_pattern_for(host, include)
        if u not in seen:
            seen.add(u)
            queue.append(u)

    for host, pref in scope.items():
        cre = content_re_of[host]
        motif = f"contenu ~ {cre.pattern!r}" if cre else "AUCUN filtre de contenu (tout suivi)"
        print(f"[scope] {host}{pref or '/'} | {motif}", file=sys.stderr)
        if cre is None:
            print(f"[info] {host} : site inconnu, aucun motif de contenu. "
                  f"Utilise --include '<regex>' pour ne garder que les pages de cours.",
                  file=sys.stderr)

    saved: list = []
    n_fetched = n_saved = n_failed = 0

    while queue and n_fetched < max_pages:
        url = queue.popleft()
        host = norm_host(urlparse(url).netloc)
        content_re = content_re_of.get(host)

        if robots is not None and not robots.allowed(url):
            print(f"[robots] ignoré {url}", file=sys.stderr)
            continue

        # 1) récupérer le HTML (via ton scraper). On récupère MÊME les pages
        #    hors-contenu (accueil, index) : elles ne sont pas enregistrées,
        #    mais leur sidebar / table des matières liste les pages de cours.
        try:
            html = scraper.fetch_page(url, session=sess)
            n_fetched += 1
        except Exception as e:
            n_failed += 1
            print(f"[erreur-fetch] {url} : {e}", file=sys.stderr)
            time.sleep(delay)
            continue

        # 2) scraper + enregistrer — UNIQUEMENT si c'est une page de contenu
        if is_content(url, content_re):
            try:
                doc = scraper.get_extractor(url).extract(html, url)
                path = scraper.save_document(doc, out_dir, url=url)
                saved.append(path)
                n_saved += 1
                print(f"[{n_fetched}/{max_pages}] ok   {url} -> {path.name}",
                      file=sys.stderr)
            except Exception as e:
                n_failed += 1
                print(f"[{n_fetched}/{max_pages}] skip {url} ({e})", file=sys.stderr)
        else:
            print(f"[{n_fetched}/{max_pages}] hub  {url} (lu pour ses liens, non enregistré)",
                  file=sys.stderr)

        # 3) découvrir de nouvelles pages (sur la page entière), en ne gardant
        #    QUE les liens de contenu de cours (dans le périmètre du site)
        pref = scope.get(host, "")
        soup = BeautifulSoup(html, "lxml")
        for link in discover_links(soup, url):
            link = norm_url(link)
            if link in seen:
                continue
            if not in_scope(link, host, pref):
                continue
            if not is_content(link, content_re):   # <-- seuls les liens de contenu
                continue
            seen.add(link)
            queue.append(link)

        time.sleep(delay)

    print(f"[terminé] récupérées={n_fetched} enregistrées={n_saved} "
          f"échecs={n_failed} découvertes={len(seen)} en_attente={len(queue)}",
          file=sys.stderr)
    return saved


def main(argv: "list[str] | None" = None) -> "list[Path]":
    """Point d'entrée CLI : crawle les seeds et sauvegarde les pages de cours.

    Args:
        argv: Arguments à analyser (défaut : ``sys.argv[1:]``).

    Returns:
        Les chemins des fichiers Markdown écrits.
    """
    ap = argparse.ArgumentParser(
        description="Crawler de site : découvre toutes les pages et les fait "
                    "scraper par web_scraper.py.")
    ap.add_argument("seeds", nargs="+",
                    help="URL(s) racine(s) à crawler (une ou plusieurs).")
    ap.add_argument("--out", default="corpus",
                    help="dossier de sortie des .md (défaut: corpus)")
    ap.add_argument("--max-pages", type=int, default=500,
                    help="arrêt après N pages récupérées (défaut: 500)")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="pause en secondes entre requêtes (défaut: 0.5)")
    ap.add_argument("--path-prefix", default=None,
                    help="force le périmètre du chemin (ex. /NYU-DLSP21). "
                         "Par défaut : déduit du seed.")
    ap.add_argument("--include", default=None,
                    help="regex sur le CHEMIN des URLs : seules les pages qui "
                         "correspondent sont suivies et enregistrées. Par défaut, "
                         "un motif connu est utilisé pour d2l.ai (/chapter_) et "
                         "atcold.github.io (/week\\d).")
    ap.add_argument("--ignore-robots", action="store_true",
                    help="ne pas respecter robots.txt (respecté par défaut)")
    a = ap.parse_args(argv)

    saved = crawl(a.seeds, a.out, max_pages=a.max_pages, delay=a.delay,
                  respect_robots=not a.ignore_robots, path_prefix=a.path_prefix,
                  include=a.include)
    print(f"\n[ok] {len(saved)} page(s) enregistrée(s) dans {a.out}/",
          file=sys.stderr)
    return saved


if __name__ == "__main__":
    main()