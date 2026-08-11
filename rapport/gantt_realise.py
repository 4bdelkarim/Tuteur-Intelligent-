import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime


# ============================================================
# 1. DONNÉES DU PROJET RÉALISÉ
# ============================================================

tasks = [
    # Nom, début, fin, catégorie
    ("1. Cadrage + État de l'art GLM-OCR",
     "2026-04-14", "2026-04-28", "glmocr"),

    ("2. Déploiement + tests GLM-OCR",
     "2026-04-21", "2026-05-10", "glmocr"),

    ("3. État de l'art RAG",
     "2026-05-11", "2026-05-22", "reorientation"),

    ("4. Conception architecture RAG",
     "2026-05-18", "2026-05-31", "reorientation"),

    ("5. Extraction PDF (GLM-OCR) + Web",
     "2026-06-01", "2026-06-21", "implementation"),

    ("6. Chunking + Embedding + Indexation",
     "2026-06-15", "2026-06-30", "implementation"),

    ("7. Retrieval + Reranking + Query Processing",
     "2026-06-23", "2026-07-12", "implementation"),

    ("8. Génération + CLI",
     "2026-07-01", "2026-07-12", "implementation"),

    ("9. Golden dataset",
     "2026-07-06", "2026-07-19", "evaluation"),

    ("10. Évaluation Ragas",
     "2026-07-14", "2026-07-27", "evaluation"),

    ("11. Rédaction du rapport",
     "2026-07-21", "2026-08-16", "redaction"),

    ("12. Soutenance",
     "2026-08-24", "2026-08-24", "soutenance"),
]


# ============================================================
# 2. COULEURS
# ============================================================

BLUE_DARK = "#203864"
BLUE_MED = "#2E5597"

colors = {
    "glmocr": "#7469D8",
    "reorientation": "#85857C",
    "implementation": "#169C78",
    "evaluation": "#D9562B",
    "redaction": "#B7770C",
    "soutenance": "#823B86",
}


# ============================================================
# 3. CONVERSION DES DATES
# ============================================================

def date(value):
    return datetime.strptime(value, "%Y-%m-%d")


# ============================================================
# 4. CRÉATION DE LA FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8))


# ============================================================
# 5. LIMITES DE L'AXE TEMPOREL
# ============================================================

start_date = date("2026-04-07")
end_date = date("2026-09-02")

ax.set_xlim(start_date, end_date)


# ============================================================
# 6. BARRES DU DIAGRAMME DE GANTT
# ============================================================

bar_height = 0.55

for i, (name, start, end, category) in enumerate(tasks):

    start = date(start)
    end = date(end)

    # Si la tâche dure 0 jour (soutenance), lui donner 1 jour
    width = max((end - start).days, 1)

    y = i + 1

    # Barre
    ax.barh(
        y=y,
        width=width,
        left=start,
        height=bar_height,
        color=colors[category],
        edgecolor="white",
        linewidth=1.2,
        zorder=3
    )


# ============================================================
# 7. PHASES (bandeaux de gauche)
# ============================================================

phase_labels = [
    (1, 2, "Phase 1\nSujet initial\nGLM-OCR"),
    (3, 4, "Phase 2\nRéorientation\nConception RAG"),
    (5, 8, "Phase 3\nImplémentation\ndu pipeline"),
    (9, 10, "Phase 4\nÉvaluation\n& Rédaction"),
]

for y_start, y_end, label in phase_labels:

    y_center = (y_start + y_end) / 2

    ax.text(
        start_date,
        y_center,
        label,
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#666666",
        style="italic"
    )


# ============================================================
# 8. JALON DE RÉORIENTATION
# ============================================================

reorientation_date = date("2026-05-15")

ax.axvline(
    reorientation_date,
    color="#D9562B",
    linestyle="--",
    linewidth=1.5,
    zorder=1
)

ax.text(
    reorientation_date,
    12.8,
    "Réorientation\nsujet",
    rotation=0,
    color="#D9562B",
    fontsize=8,
    fontweight="bold",
    ha="center",
    va="bottom"
)


# ============================================================
# 9. JALON SOUTENANCE
# ============================================================

soutenance_date = date("2026-08-24")

ax.axvline(
    soutenance_date,
    color="#823B86",
    linestyle="-",
    linewidth=2.0,
    zorder=4
)

ax.scatter(
    soutenance_date,
    12,
    s=80,
    color="#823B86",
    zorder=5,
    marker="D"
)


# ============================================================
# 10. AXE Y
# ============================================================

yticks = list(range(1, 13))
yticklabels = [t[0] for t in tasks]

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=9)
ax.set_ylim(13.2, 0.3)


# ============================================================
# 11. SEMAINES
# ============================================================

week_positions = [
    date("2026-04-14"), date("2026-04-21"), date("2026-04-28"),
    date("2026-05-05"), date("2026-05-12"), date("2026-05-19"),
    date("2026-05-26"), date("2026-06-02"), date("2026-06-09"),
    date("2026-06-16"), date("2026-06-23"), date("2026-06-30"),
    date("2026-07-07"), date("2026-07-14"), date("2026-07-21"),
    date("2026-07-28"), date("2026-08-04"), date("2026-08-11"),
    date("2026-08-18"), date("2026-08-25"),
]

# Lignes verticales
for d in week_positions:
    ax.axvline(d, color="#E8E8E8", linewidth=0.6, zorder=0)


# ============================================================
# 12. MOIS
# ============================================================

month_starts = [
    date("2026-04-01"), date("2026-05-01"), date("2026-06-01"),
    date("2026-07-01"), date("2026-08-01"), date("2026-09-01"),
]

month_names = ["Avril", "Mai", "Juin", "Juillet", "Août", "Sept."]

for month_date, month_name in zip(month_starts, month_names):

    ax.axvline(month_date, color="#CCCCCC", linewidth=1.2, zorder=1)

    ax.text(
        month_date + (date("2026-04-08") - date("2026-04-01")),
        0.1,
        month_name,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#555555"
    )


# ============================================================
# 13. AXE DES DATES
# ============================================================

ax.xaxis.set_major_locator(
    mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", length=0)


# ============================================================
# 14. GRILLE
# ============================================================

ax.grid(axis="x", color="#F0F0F0", linewidth=0.6)
ax.set_axisbelow(True)


# ============================================================
# 15. SUPPRESSION DES BORDURES
# ============================================================

for spine in ax.spines.values():
    spine.set_visible(False)


# ============================================================
# 16. LÉGENDE
# ============================================================

legend_elements = [
    Patch(facecolor=colors["glmocr"], label="GLM-OCR"),
    Patch(facecolor=colors["reorientation"], label="Réorientation"),
    Patch(facecolor=colors["implementation"], label="Implémentation"),
    Patch(facecolor=colors["evaluation"], label="Évaluation"),
    Patch(facecolor=colors["redaction"], label="Rédaction"),
    Patch(facecolor=colors["soutenance"], label="Soutenance"),
]

legend = ax.legend(
    handles=legend_elements,
    loc="lower right",
    ncol=6,
    frameon=True,
    fontsize=9,
    handlelength=1.2,
    columnspacing=1.0,
    bbox_to_anchor=(1.0, -0.18),
)

legend.get_frame().set_edgecolor("#DDDDDD")
legend.get_frame().set_linewidth(0.5)


# ============================================================
# 17. TITRE
# ============================================================

fig.suptitle(
    "Planning réalisé — Tuteur Pédagogique RAG",
    fontsize=18,
    fontweight="bold",
    color=BLUE_DARK,
    y=0.97
)

fig.text(
    0.5,
    0.935,
    "Avril → Août 2026  |  Soutenance le 24 août  |  LIMOS — ISIMA",
    ha="center",
    fontsize=10,
    style="italic",
    color="#666666"
)


# ============================================================
# 18. EXPORT
# ============================================================

plt.savefig(
    "images/gantt_realise.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "images/gantt_realise.pdf",
    bbox_inches="tight",
    facecolor="white"
)
