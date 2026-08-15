# Tuteur Pédagogique RAG — v2

Système de tutorat pédagogique intelligent fondé sur une architecture RAG
(Retrieval-Augmented Generation), appliqué à un corpus de cours de machine learning.

## Architecture

```
src/rag_tutor/
├── core/           ← Pipeline RAG (retrieval, génération, refus)
├── extraction/     ← Extraction PDF (GLM-OCR) + Web
├── ingestion/      ← Indexation ChromaDB
├── evaluation/     ← Métriques Ragas + hit@k/MRR
├── conversation/   ← Mémoire conversationnelle + prompts socratiques
└── cli/            ← Interfaces (chat Q&A, tuteur conversationnel)
```

## Installation

```bash
pip install -e .
```

## Utilisation

```bash
# Tuteur conversationnel (mode socratique)
rag-tutor

# Questions/réponses directes
rag-chat --show-sources

# Évaluation
rag-eval eval/test_set_v2.json --retrieval-mode hybrid_rerank
```

## Données

- `data/raw/` — sources brutes (PDFs de cours + markdown web)
- `data/processed/` — corpus unifié au format canonique (165+ fichiers .md)
- `dbfig_pc/` — index ChromaDB prêt à l'emploi (collection `cours_ml_fig`)

Pour réindexer après modification du corpus : `make ingest DIR=data/processed`.

## Prérequis

- Ollama avec les modèles : `qwen2.5:14b`, `qwen3:8b`, `bge-m3`
- GPU recommandé (H100 NVL testé, ~95 Go VRAM)
- GLM-OCR pour l'extraction PDF

## Rapport

Le rapport de stage vit dans un dépôt séparé : `ai-tutor-rapport`.
