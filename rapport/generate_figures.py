#!/usr/bin/env python3
"""Génère les 3 figures PNG pour le chapitre 7 du rapport."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Couleurs ISIMA
BLEU = '#1F3864'
BLEU_CLAIR = '#2E5597'
GRIS = '#404040'
ORANGE = '#E87722'
VERT = '#2E7D32'
FOND = '#F5F7FA'

OUT = '/mnt/data/abdelkarim/rag-tutor-v2/rapport/images'

# ============================================================
# FIGURE 1 : Flux conversationnel complet
# ============================================================
def figure_flux():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def box(x, y, w, h, text, color=BLEU, fontsize=9, text_color='white', bold=False):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.15", linewidth=1.2,
                              facecolor=color, edgecolor='white')
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, weight=weight)

    def arrow(x1, y1, x2, y2, color=GRIS):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    def label(x, y, text, fontsize=8, color=GRIS):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=color, style='italic')

    # ÉTUDIANT
    box(0.8, 8.5, 1.8, 0.8, 'Étudiant', BLEU, 11, bold=True)
    arrow(1.7, 8.5, 3.3, 8.5)
    label(2.5, 9.0, 'question')

    # TUTOR.PY
    box(4.5, 8.5, 2.2, 0.8, 'tutor.py', BLEU_CLAIR, 10, bold=True)
    arrow(5.6, 8.1, 5.6, 6.8)
    label(4.0, 7.5, 'memory.add_turn\n("student")')
    label(6.2, 7.5, 'memory.get_\nformatted_history()')

    # PIPELINE.ANSWER()
    box(5.6, 6.3, 2.2, 0.8, 'pipeline.answer()', VERT, 10, bold=True)

    # 3 composants
    arrow(4.5, 5.9, 2.0, 4.5)
    arrow(5.6, 5.9, 5.6, 4.5)
    arrow(6.7, 5.9, 9.0, 4.5)

    box(2.0, 4.0, 2.2, 0.9, 'QP contextuel\n(anaphores)', BLEU, 8)
    box(5.6, 4.0, 2.2, 0.9, 'Retriever\n(hybride)', BLEU, 8)
    box(9.0, 4.0, 2.2, 0.9, 'Generator\n(streaming)', BLEU, 8)

    # Flèches de retour vers tutor.py
    arrow(9.0, 3.5, 7.5, 2.0)
    arrow(5.6, 3.5, 5.6, 2.0)
    label(6.8, 2.3, 'tokens')
    label(4.0, 2.5, 'documents')
    arrow(2.0, 3.5, 3.5, 2.0)

    # RÉPONSE
    box(5.6, 1.5, 3.0, 0.8, 'Réponse token/token', ORANGE, 10, bold=True)
    arrow(5.6, 1.1, 5.6, 0.3)
    box(5.6, 0.0, 2.5, 0.7, '→ Étudiant', BLEU, 9, bold=True)

    # COMPRESSION
    label(1.8, 1.5, 'memory.add_turn\n("tutor")')
    label(3.8, 1.5, 'memory.compress()')
    arrow(7.0, 1.5, 9.0, 1.5)
    box(9.5, 1.5, 1.5, 0.8, 'si > 6\ntours', FOND, 7, text_color=GRIS)

    ax.set_title('Flux conversationnel — Tuteur v2', fontsize=14, color=BLEU,
                 weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_flux_conversationnel.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ fig_flux_conversationnel.png')

# ============================================================
# FIGURE 2 : Stratégie de compression mémoire
# ============================================================
def figure_compression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor('white')

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    titles = ['AVANT compression (8 tours)', 'APRÈS compression']
    for ax, title in zip([ax1, ax2], titles):
        ax.set_title(title, fontsize=12, color=BLEU, weight='bold', pad=10)

    def tour_box(ax, x, y, w, h, role, text, color, alpha=1.0):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.08", linewidth=0.8,
                              facecolor=color, edgecolor='white', alpha=alpha)
        ax.add_patch(rect)
        ax.text(x - w/2 + 0.15, y + 0.15, role, fontsize=7, color='white',
                weight='bold', va='center')
        ax.text(x, y - 0.08, text, fontsize=6.5, color='white', ha='center',
                va='center')

    def ancien_label(ax, x, y, text, fontsize=7):
        ax.text(x, y, text, fontsize=fontsize, color=BLEU, weight='bold',
                ha='left', style='italic')

    # --- AVANT ---
    ax1.text(0.5, 9.3, 'Tours 1-2 (anciens)', fontsize=8, color=ORANGE,
             weight='bold')
    tour_box(ax1, 5, 8.5, 9, 1.2, 'É', "c'est quoi une LSTM ?", BLEU_CLAIR, 0.4)
    tour_box(ax1, 5, 7.7, 9, 1.2, 'T', "Une LSTM est un type de RNN avec\nporte d'oubli, d'entrée, de sortie...", BLEU_CLAIR, 0.4)

    # Flèche vers résumé
    ax1.annotate('', xy=(9.5, 8.1), xytext=(9.8, 8.1),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))
    ax1.text(9.3, 7.5, 'résumés', fontsize=7, color=ORANGE, ha='center',
             style='italic')

    ax1.text(0.5, 6.8, 'Tours 3-8 (récents, intacts)', fontsize=8, color=VERT,
             weight='bold')
    for i, (role, txt) in enumerate([
        ('É', 'comment elle résout le\nvanishing gradient ?'),
        ('T', 'Les portes permettent de\ncontrôler le flux...'),
        ('É', 'compare-la avec le CNN'),
        ('T', 'Les CNN utilisent des filtres\nconvolutifs...'),
    ]):
        y = 5.8 - i * 1.1
        tour_box(ax1, 5, y, 9, 0.9, role, txt, BLEU_CLAIR)

    # --- APRÈS ---
    # Résumé box
    ax2.text(0.5, 9.3, 'RÉSUMÉ', fontsize=9, color=ORANGE, weight='bold')
    rect = FancyBboxPatch((0.3, 8.0), 9.4, 1.5,
                          boxstyle="round,pad=0.15", linewidth=1.2,
                          facecolor=ORANGE, edgecolor='white', alpha=0.15)
    ax2.add_patch(rect)
    ax2.text(5, 8.75, "L'étudiant a exploré les LSTM, leurs portes (oubli,\n"
                       "entrée, sortie) et la différence avec les RNN classiques.",
             fontsize=8, color=ORANGE, ha='center', va='center',
             style='italic')

    ax2.text(0.5, 7.2, 'Tours 3-8 (intacts)', fontsize=8, color=VERT,
             weight='bold')
    for i, (role, txt) in enumerate([
        ('É', 'comment elle résout le\nvanishing gradient ?'),
        ('T', 'Les portes permettent de\ncontrôler le flux...'),
        ('É', 'compare-la avec le CNN'),
        ('T', 'Les CNN utilisent des filtres\nconvolutifs...'),
    ]):
        y = 6.2 - i * 1.1
        tour_box(ax2, 5, y, 9, 0.9, role, txt, BLEU_CLAIR)

    fig.suptitle('Stratégie de compression de la mémoire conversationnelle',
                 fontsize=14, color=BLEU, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_compression_memoire.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ fig_compression_memoire.png')

# ============================================================
# FIGURE 3 : Impact du QP contextuel
# ============================================================
def figure_qp():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    metrics = ['Overlap lexical', "Requête\nreformulée", 'Temps génération\n(s)', 'Score reranker\n(top)']
    sans_qp = [0, 0, 8.6, 0.488]
    avec_qp = [100, 100, 12.8, 0.491]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, sans_qp, width, label='Sans QP contextuel',
                   color=BLEU_CLAIR, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, avec_qp, width, label='Avec QP contextuel',
                   color=VERT, alpha=0.85, edgecolor='white')

    # Normalisation : les 2 premières métriques sont en %, les 2 dernières en valeurs absolues
    ax.set_ylabel('Valeur', fontsize=10, color=GRIS)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9, color=GRIS)
    ax.legend(fontsize=9, loc='upper left')

    # Annotations
    def annotate(bar, val, fmt, offset_y=0.5):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_y,
                fmt.format(val), ha='center', va='bottom', fontsize=8, color=GRIS)

    for bar, val in zip(bars1, sans_qp):
        if val < 1:
            annotate(bar, val, '{:.3f}', 0.3)
        elif val > 10:
            annotate(bar, val, '{:.1f}', 0.3)
        else:
            annotate(bar, val, '{}', 0.3)

    for bar, val in zip(bars2, avec_qp):
        if val < 1:
            annotate(bar, val, '{:.3f}', 0.3)
        elif val > 10:
            annotate(bar, val, '{:.1f}', 0.3)
        else:
            annotate(bar, val, '{}', 0.3)

    # Annotation spéciale pour l'overlap
    ax.annotate('+100 pts', xy=(0.35, 105), fontsize=9, color=VERT, weight='bold',
                ha='center')
    ax.annotate('Anaphore\nrésolue', xy=(1.35, 108), fontsize=8, color=VERT,
                ha='center', style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 120)
    ax.set_title('Impact du QP contextuel (Q2 : « compare-la avec le CNN »)',
                 fontsize=12, color=BLEU, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_qp_contextuel.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ fig_qp_contextuel.png')


if __name__ == '__main__':
    figure_flux()
    figure_compression()
    figure_qp()
    print('\n✅ 3 figures générées dans rapport/images/')
