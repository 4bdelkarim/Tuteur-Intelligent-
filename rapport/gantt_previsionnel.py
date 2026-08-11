import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime


# ============================================================
# 1. DONNÉES DU PROJET
# ============================================================

tasks = [
    # Nom, début, fin, catégorie, texte dans la barre
    ("1. Cadrage + État de l'art",
     "2026-04-08", "2026-04-19", "cadrage", ""),

    ("2. Architecture GLM-OCR",
     "2026-04-19", "2026-04-30", "architecture", "S2-S3"),

    ("3. Appropriation du code",
     "2026-05-01", "2026-05-10", "implementation", "S4-S5"),

    ("4. Déploiement Ollama",
     "2026-05-11", "2026-05-24", "implementation", "S6-S7"),

    ("5. Protocole + corpus",
     "2026-05-25", "2026-06-07", "experimentation", "S8-S9"),

    ("6. Expérimentations",
     "2026-06-08", "2026-07-05", "experimentation", "S10-S13"),

    ("7. Rédaction finale",
     "2026-07-06", "2026-07-24", "redaction", "S14-S16"),

    ("8. API (secondaire)",
     "2026-07-06", "2026-07-19", "api", "si temps"),

    ("9. Soutenance",
     "2026-08-24", "2026-08-31", "soutenance", "Août"),
]


# ============================================================
# 2. COULEURS
# ============================================================

colors = {
    "cadrage": "#85857C",
    "architecture": "#7469D8",
    "implementation": "#169C78",
    "experimentation": "#D9562B",
    "redaction": "#B7770C",
    "api": "#DDA52A",
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

fig = plt.figure(figsize=(16, 10))

# Zone principale du diagramme de Gantt
ax = fig.add_axes([0.27, 0.50, 0.67, 0.40])


# ============================================================
# 5. LIMITES DE L'AXE TEMPOREL
# ============================================================

start_date = date("2026-04-08")
end_date = date("2026-08-31")

ax.set_xlim(start_date, end_date)


# ============================================================
# 6. POSITION DES TÂCHES SUR L'AXE Y
# ============================================================

task_y = {}

for i, task in enumerate(tasks):
    task_y[task[0]] = i + 1


# ============================================================
# 7. BARRES DU DIAGRAMME DE GANTT
# ============================================================

bar_height = 0.55

for i, (name, start, end, category, label) in enumerate(tasks):

    start = date(start)
    end = date(end)

    y = i + 1

    # Barre
    ax.barh(
        y=y,
        width=(end - start).days,
        left=start,
        height=bar_height,
        color=colors[category],
        edgecolor="white",
        linewidth=1.2,
        zorder=3
    )

    # Texte à l'intérieur de la barre
    if label:

        text_color = "white"

        # Pour API, texte foncé
        if category == "api":
            text_color = "#4F3B00"

        ax.text(
            start + (end - start) / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text_color,
            zorder=4
        )


# ============================================================
# 8. RÉDACTION CONTINUE
# ============================================================

# Position sous les 9 tâches
y_redaction = 10

redaction_start = date("2026-04-19")
redaction_end = date("2026-07-12")

ax.barh(
    y_redaction,
    (redaction_end - redaction_start).days,
    left=redaction_start,
    height=0.45,
    color="#CFCFCF",
    edgecolor="none",
    zorder=2
)

ax.text(
    redaction_start + (redaction_end - redaction_start) / 2,
    y_redaction,
    "en parallèle des phases 2 à 6",
    ha="center",
    va="center",
    fontsize=9,
    color="#666666",
    style="italic"
)


# ============================================================
# 9. RÉUNIONS ENCADRANT
# ============================================================

y_meeting = 11

meetings = [
    "2026-04-19",
    "2026-04-30",
    "2026-05-10",
    "2026-05-24",
    "2026-06-07",
    "2026-06-21",
    "2026-07-05",
    "2026-07-24",
]

for meeting in meetings:

    d = date(meeting)

    ax.scatter(
        d,
        y_meeting,
        s=30,
        color="#5149B8",
        zorder=5
    )


# ============================================================
# 10. DEADLINE DU 30/04
# ============================================================

deadline = date("2026-04-30")

ax.axvline(
    deadline,
    color="red",
    linestyle="--",
    linewidth=1.3,
    zorder=1
)

ax.text(
    deadline,
    5.5,
    "Deadline 30/04",
    rotation=90,
    color="red",
    fontsize=8,
    ha="right",
    va="center"
)


# ============================================================
# 11. AXE Y
# ============================================================

yticks = list(range(1, 10)) + [y_redaction, y_meeting]

yticklabels = [
    "1. Cadrage + État de l'art",
    "2. Architecture GLM-OCR",
    "3. Appropriation du code",
    "4. Déploiement Ollama",
    "5. Protocole + corpus",
    "6. Expérimentations",
    "7. Rédaction finale",
    "8. API (secondaire)",
    "9. Soutenance",
    "Rédaction continue",
    "Réunions encadrant"
]

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=10)

# IMPORTANT :
# Matplotlib place naturellement les petites valeurs en bas.
# On inverse uniquement les limites numériques de l'axe,
# PAS les données ni l'ordre des tâches.
ax.set_ylim(11.5, 0.3)


# ============================================================
# 12. SEMAINES S1 → S16
# ============================================================

week_positions = [
    date("2026-04-08"),
    date("2026-04-15"),
    date("2026-04-22"),
    date("2026-04-30"),
    date("2026-05-07"),
    date("2026-05-14"),
    date("2026-05-21"),
    date("2026-05-28"),
    date("2026-06-04"),
    date("2026-06-11"),
    date("2026-06-18"),
    date("2026-06-25"),
    date("2026-07-02"),
    date("2026-07-09"),
    date("2026-07-16"),
    date("2026-07-24"),
]

# Lignes verticales
for d in week_positions:

    ax.axvline(
        d,
        color="#D9D9D9",
        linewidth=0.8,
        zorder=0
    )


# Labels S1 à S16
for i, d in enumerate(week_positions, start=1):

    ax.text(
        d,
        0.05,
        f"S{i}",
        ha="center",
        va="bottom",
        fontsize=9
    )


# ============================================================
# 13. AXE DES DATES
# ============================================================

ax.xaxis.set_major_locator(
    mdates.WeekdayLocator(
        byweekday=mdates.MO,
        interval=1
    )
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d/%m")
)

ax.tick_params(
    axis="x",
    labelsize=8
)

ax.tick_params(
    axis="y",
    length=0
)


# ============================================================
# 14. GRILLE
# ============================================================

ax.grid(
    axis="x",
    color="#E5E5E5",
    linewidth=0.7
)

ax.set_axisbelow(True)


# ============================================================
# 15. SUPPRESSION DES BORDURES
# ============================================================

for spine in ax.spines.values():
    spine.set_visible(False)


# ============================================================
# 16. TITRE
# ============================================================

fig.suptitle(
    "Planning TER GLM-OCR — v2",
    fontsize=20,
    fontweight="bold",
    color="#203864",
    y=0.955
)

fig.text(
    0.61,
    0.925,
    "08/04/2026 → 24/07/2026  |  Soutenance fin août  |  Salle B013  |  35h/sem",
    ha="center",
    fontsize=11,
    style="italic"
)


# ============================================================
# 17. JALONS ET LIVRABLES CLÉS
# ============================================================

fig.text(
    0.05,
    0.40,
    "Jalons et livrables clés",
    fontsize=14,
    fontweight="bold",
    color="#203864"
)

milestones = [
    "S1 (19/04) — État de l'art v1 transmis",
    "S3 (30/04) — Architecture GLM-OCR maîtrisée (deadline ferme)",
    "S5 (10/05) — Code source approprié + env. installé",
    "S7 (24/05) — GLM-OCR fonctionnel en local",
    "S9 (07/06) — Corpus de test + protocole d'évaluation",
    "S13 (05/07) — Résultats expérimentaux + analyse",
    "S16 (24/07) — Rapport final remis (priorité absolue)",
    "Fin août — Soutenance"
]

y = 0.365

for text in milestones:

    fig.text(
        0.055,
        y,
        "•",
        fontsize=15,
        color="#5149B8"
    )

    fig.text(
        0.07,
        y,
        text,
        fontsize=10
    )

    y -= 0.027


# ============================================================
# 18. PRIORISATION
# ============================================================

fig.text(
    0.05,
    0.135,
    "Priorisation",
    fontsize=14,
    fontweight="bold",
    color="#203864"
)

fig.text(
    0.05,
    0.100,
    "Priorité 1 (coeur scientifique) — phases 1 à 7 : "
    "état de l'art, architecture, implémentation, évaluation, "
    "rédaction. Non négociables.",
    fontsize=10
)

fig.text(
    0.05,
    0.067,
    "Priorité 2 (bonus) — phase 8 : API REST pour interroger le modèle. "
    "Développée uniquement si le coeur scientifique est solidement terminé.",
    fontsize=10
)


# ============================================================
# 19. POINTS À CLARIFIER
# ============================================================

fig.text(
    0.48,
    0.40,
    "△ Points à clarifier rapidement",
    fontsize=14,
    fontweight="bold",
    color="#203864"
)

fig.text(
    0.48,
    0.365,
    "Corpus de documents cibles — à trancher avec l'encadrant avant S5. "
    "Impact direct sur les tests d'évaluation.",
    fontsize=10
)

fig.text(
    0.48,
    0.330,
    "Contrainte matérielle (GPU / CPU) — à trancher avant S6. "
    "Impact sur les temps de calcul et l'approche d'évaluation.",
    fontsize=10
)


# ============================================================
# 20. LÉGENDE
# ============================================================

fig.text(
    0.48,
    0.285,
    "Légende",
    fontsize=14,
    fontweight="bold",
    color="#203864"
)

legend_elements = [
    Patch(
        facecolor=colors["cadrage"],
        label="Cadrage"
    ),

    Patch(
        facecolor=colors["architecture"],
        label="Analyse archi."
    ),

    Patch(
        facecolor=colors["redaction"],
        label="Rédaction"
    ),

    Patch(
        facecolor=colors["soutenance"],
        label="Soutenance"
    ),

    Patch(
        facecolor=colors["implementation"],
        label="Implémentation"
    ),

    Patch(
        facecolor=colors["experimentation"],
        label="Expérimentation"
    ),

    Patch(
        facecolor=colors["api"],
        label="API (bonus)"
    ),
]

ax_legend = fig.add_axes(
    [0.48, 0.16, 0.35, 0.11]
)

ax_legend.axis("off")

ax_legend.legend(
    handles=legend_elements,
    loc="upper left",
    ncol=2,
    frameon=False,
    fontsize=10,
    handlelength=1.0,
    columnspacing=2.0
)

# Jalon / réunion
fig.text(
    0.755,
    0.205,
    "•",
    fontsize=16,
    color="#5149B8"
)

fig.text(
    0.775,
    0.205,
    "Jalon / réunion",
    fontsize=10
)


# ============================================================
# 21. EXPORT PNG
# ============================================================

plt.savefig(
    "images/gantt_previsionnel.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ============================================================
# 22. EXPORT PDF
# ============================================================

plt.savefig(
    "images/gantt_previsionnel.pdf",
    bbox_inches="tight",
    facecolor="white"
)
