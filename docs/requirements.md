# Requirements — Tuteur RAG Pédagogique (`rag-tutor v2.0.0`)

> **Document de référence** pour reproduire l'environnement complet du pipeline.

---

## 1. Résumé exécutif

Le tuteur RAG est un pipeline complet qui ingère des PDFs de cours + pages web, les transforme en markdown via GLM-OCR + Qwen3-VL (figures), les indexe dans ChromaDB, puis expose un moteur de Q&A conversationnel avec :
- Retrieval hybride (BM25 + dense + cross-encoder reranker)
- Mécanisme de refus double (M1 pré-génération + M2 post-génération)
- Query processing (reformulation + décomposition)
- Conversation memory (mode tuteur socratique)

Tout tourne **100% localement** (Ollama + ChromaDB), sans aucun appel API externe (hors téléchargement initial des modèles).

---

## 2. Matériel (hardware)

| Composant | Minimum | Recommandé | Notes |
|-----------|---------|------------|-------|
| **RAM** | 32 GB | 64 GB | Les modèles LLM chargent en mémoire CPU si la VRAM est insuffisante |
| **VRAM GPU** | 16 GB | 24+ GB | Pour charger tous les modèles runtime (sans OCR/VL) |
| **VRAM GPU (tout)** | 32 GB | 48+ GB | Si OCR PDF + description figures exécutés en parallèle |
| **Stockage disque** | 50 GB | 100 GB | Modèles Ollama (~30 GB) + corpus + index + datasets |
| **CPU** | 8 cœurs | 16+ cœurs | L'embedding BGE-M3 et le cross-encoder tournent sur CPU si GPU indisponible |

### Détail VRAM par modèle

| Modèle | VRAM | Disque | Rôle | Exécution |
|--------|------|--------|------|-----------|
| `qwen2.5:14b` | ~9.0 GB | ~8.5 GB | Génération + juge évaluation | Runtime |
| `qwen3:8b` | ~5.5 GB | ~5.0 GB | Vérification post-génération (M2) | Runtime |
| `bge-m3` | ~1.5 GB | ~2.2 GB | Embeddings denses + sparse | Runtime |
| `bge-reranker-v2-m3` | ~0.7 GB | ~2.2 GB | Cross-encoder reranking | Runtime |
| `glm-ocr` | ~11.0 GB | ~7.0 GB | OCR des PDF | Étape séparée (ingestion) |
| `qwen3-vl:8b` | ~6.0 GB | ~5.5 GB | Description des figures | Étape séparée (ingestion) |
| **Total runtime** | **~16.7 GB** | **~17.9 GB** | Sans OCR ni VLM | |
| **Recommandé (avec marge KV-cache)** | **~25 GB** | — | | |
| **Total tous modèles** | **~33.7 GB** | **~30.4 GB** | | |

---

## 3. Logiciel (software)

### 3.1 Système d'exploitation

- **Linux** (Ubuntu 22.04+ recommandé) — l'index PyTorch CUDA cible `cu126` (CUDA 12.6)
- macOS (Apple Silicon) fonctionne aussi (Ollama + ChromaDB natifs) mais PyTorch CUDA non disponible
- **Python ≥ 3.11**

### 3.2 Services obligatoires

| Service | Version | Rôle | Commande de démarrage |
|---------|---------|------|----------------------|
| **Ollama** | ≥ 0.4.0 | Serveur d'inférence LLM local | `ollama serve` (port 11434) |
| **ChromaDB** | ≥ 0.5.0 | Base vectorielle persistante | Pas de serveur — bibliothèque embarquée |

Ollama doit être configuré avec `OLLAMA_HOST=http://127.0.0.1:11434` (valeur par défaut dans `.env`).

### 3.3 Modèles Ollama à télécharger

```bash
# Modèles runtime (obligatoires)
ollama pull qwen2.5:14b          # Génération principale + évaluation
ollama pull qwen3:8b             # Vérification post-génération (M2)
ollama pull bge-m3               # Embeddings denses + sparse

# Modèles ingestion (optionnels, exécutés séparément)
ollama pull glm-ocr:latest       # OCR PDF (GLM-OCR via Ollama)
ollama pull qwen3-vl:8b          # Description de figures (vision)
```

**Note :** `bge-reranker-v2-m3` est le seul modèle qui ne passe PAS par Ollama. Il est téléchargé automatiquement par `sentence-transformers` depuis HuggingFace (~600 Mo, une seule fois).

### 3.4 Dépendances Python

```toml
# Extrait de pyproject.toml
dependencies = [
    "glmocr[selfhosted]>=0.1.5",    # OCR PDF natif + scanné
    "torch>=2.13.0",                 # Backend ML (CUDA 12.6)
    "chromadb>=0.5.0",              # Base vectorielle
    "sentence-transformers>=3.0.0", # Cross-encoder reranker
    "rank-bm25>=0.2.2",            # BM25 (sparse retrieval)
    "pymupdf>=1.24.0",             # Lecture PDF
    "ollama>=0.4.0",               # Client Ollama (chat + embeddings)
    "ragas>=0.3.0",                # Évaluation RAG
    "langchain-ollama>=0.2.0",     # Évaluation via LangChain/Ragas
    "deepeval>=1.0.0",             # Génération golden dataset
    "pyyaml>=6.0",                 # Configs + front-matter
    "tqdm>=4.66.0",                # Barres de progression
    "requests>=2.31.0",            # HTTP (crawler, APIs)
]
```

Installation :
```bash
pip install -e .
```

### 3.5 Arbre des dépendances (directes ★ + transitives)

Les 13 dépendances directes entraînent ~60 packages transitifs. Voici l'arbre complet regroupé par rôle dans le pipeline.

```
★ = dépendance directe (pyproject.toml)
  = transitive (tirée automatiquement)
```

#### Extraction PDF

```
★ glmocr 0.1.5               → pillow, numpy, requests, pydantic, PyYAML, python-dotenv, tqdm
★ pymupdf 1.28.0              → (feuille, 0 transitive)
```

#### Embeddings + Reranking

```
★ sentence-transformers 5.6.0 → transformers, huggingface-hub, torch, numpy, scikit-learn, scipy, tqdm
  └─ transformers 5.14.1      → tokenizers, huggingface-hub, numpy, pyyaml, safetensors
  └─ huggingface-hub 1.16.1   → (cache des modèles, téléchargement BGE-reranker)
  └─ tokenizers 0.22.2        → huggingface-hub
★ torch 2.13.0 (CUDA 12.6)    → numpy, sympy, networkx, jinja2, fsspec, filelock
  └─ numpy 2.4.6              → (feuille)
  └─ scipy 1.17.1             → numpy
  └─ scikit-learn 1.9.0       → numpy, scipy, joblib
★ rank-bm25 0.2.2             → numpy (BM25 sparse retrieval)
```

#### Base vectorielle

```
★ chromadb 1.5.9              → onnxruntime, pydantic, numpy, uvicorn, posthog, ...
  └─ onnxruntime 1.27.0       → protobuf, flatbuffers, numpy
  └─ protobuf 7.35.1          → (feuille)
```

#### Client LLM

```
★ ollama 0.6.2                → httpx, pydantic
  └─ httpx 0.28.1             → certifi, httpcore, idna
  └─ pydantic 2.13.4          → annotated-types, pydantic-core, typing-extensions
```

#### Évaluation

```
★ ragas 0.4.3                 → datasets, tiktoken, numpy, pydantic, typer
  └─ datasets 5.0.0           → huggingface-hub, pandas, pyarrow, requests, tqdm
★ deepeval 4.1.1              → grpcio, openai, opentelemetry-api, opentelemetry-sdk, jinja2, ...
  └─ grpcio 1.82.1            → (gRPC pour communication interne)
  └─ opentelemetry-api 1.44.0 → (tracing/monitoring)
  └─ kubernetes 36.0.3        → (⚠️ tiré par DeepEval, non utilisé)
★ langchain-ollama 1.1.0      → langchain-core, ollama
  └─ langchain-core 1.4.9     → langsmith, pydantic, pyyaml, tenacity
```

#### Utilitaires

```
★ PyYAML 6.0.3                → (feuille, 0 transitive)
★ tqdm 4.69.0                 → (feuille)
★ requests 2.34.2             → certifi, urllib3, idna, charset_normalizer
```

#### Poids total

| Catégorie | Packages | Poids estimé |
|-----------|----------|-------------|
| ML core (`torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `scipy`, `numpy`) | ~12 | **~3-4 Go** |
| Évaluation (`ragas`, `deepeval`, `datasets`, `langchain-*`, `grpcio`, `opentelemetry`, `kubernetes`) | ~20 | ~500 Mo |
| Base vectorielle (`chromadb`, `onnxruntime`, `protobuf`) | ~8 | ~200 Mo |
| PDF / OCR (`glmocr`, `pymupdf`) | ~6 | ~100 Mo |
| Léger (`ollama`, `rank-bm25`, `pyyaml`, `tqdm`, `requests`, `httpx`, `pydantic`) | ~10 | ~30 Mo |
| **Total environnement pip** | **~60** | **~4-5 Go** |

#### ⚠️ Points d'attention

| Package | Poids | Remarque |
|---------|-------|----------|
| `torch` (CUDA 12.6) | ~2.5 Go | Le plus lourd. Index PyTorch dédié (`cu126`), pas PyPI |
| `deepeval` | ~200 Mo | Tire `kubernetes`, `opentelemetry`, `grpcio` — lourd pour un simple évaluateur |
| `datasets` (via `ragas`) | ~100 Mo | Tire `pandas`, `pyarrow` — utile seulement pour l'évaluation |
| `onnxruntime` (via `chromadb`) | ~150 Mo | Runtime ONNX embarqué dans ChromaDB |

> **Reproductibilité** : le fichier `uv.lock` fige toutes les versions exactes (directes + transitives).
> Pour reproduire l'environnement à l'identique : `uv sync`

---

## 4. Configuration

### 4.1 Variables d'environnement (`.env`)

```bash
# Serveur Ollama
OLLAMA_HOST=http://127.0.0.1:11434

# Persistance ChromaDB
CHROMA_DB_PATH=data/chroma_db

# Modèles (surcharge possible)
GEN_MODEL=qwen2.5:14b
JUDGE_MODEL=qwen3:8b
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 4.2 Configuration GLM-OCR (`config/glmocr_config.yaml`)

- Mode **self-hosted** via Ollama (`api_mode: ollama_generate`, port 11434)
- Layout model : `PaddlePaddle/PP-DocLayoutV3_safetensors` (CPU, batch=1)
- `max_workers: 2`, `request_timeout: 120s`
- Sortie : JSON + markdown

### 4.3 Paramètres internes (codés en dur dans `core/`)

| Paramètre | Valeur | Module | Signification |
|-----------|--------|--------|---------------|
| `GEN_MODEL` | `qwen2.5:14b` | `llm_client.py` | Modèle de génération |
| `MAX_TOKENS` | 900 | `llm_client.py` | Tokens max par réponse |
| `NUM_CTX` | 24576 | `llm_client.py` | Fenêtre de contexte (tokens) |
| `temperature` | 0.2 | `llm_client.py` | Température de génération |
| `CHILD_TARGET` | 400 car. | `chunking.py` | Taille cible des enfants |
| `CHILD_MAX` | 750 car. | `chunking.py` | Taille max des enfants |
| `CHILD_OVERLAP` | 80 car. | `chunking.py` | Chevauchement entre enfants |
| `BM25_K` | 20 | `retriever.py` | Candidats BM25 |
| `DENSE_K` | 20 | `retriever.py` | Candidats denses |
| `RERANK_CANDIDATES` | 40 | `retriever.py` | Paires passées au reranker |
| `FINAL_CHILDREN` | 18 | `retriever.py` | Enfants gardés après rerank |
| `RRF_K` | 60 | `retriever.py` | Constante RRF |
| `RRF_ALPHA` | 0.6 | `retriever.py` | Pondération dense vs BM25 |
| `MODE` | `hybrid_rerank` | `retriever.py` | Chaîne de retrieval active |
| `RERANKER_THRESHOLD` | 0.1119 | `refusal_gate.py` | Seuil de refus M1 (calibré) |
| `CONFIDENCE_THRESHOLD` | 2 | `refusal_gate.py` | Seuil de refus M2 (confiance) |
| `DB_DIR` | `dbfig_pc` | `vector_store.py` | Dossier ChromaDB |
| `COLLECTION_NAME` | `cours_ml_fig` | `vector_store.py` | Nom de la collection |

---

## 5. Données (data)

### 5.1 Structure

```
data/
├── raw/                       ← Sources brutes (PDF + URLs)
│   ├── pdf/                   ← 8 fichiers PDF (~29 Mo)
│   └── web/                   ← 2 fichiers web bruts (~3.7 Mo)
├── processed/                 ← Markdown unifié après extraction
│   └── *.md                   ← 165+ fichiers (format canonique)
└── chroma_db/                 ← Index ChromaDB (~106 Mo)
    └── chroma.sqlite3         ← Base vectorielle persistante
```

### 5.2 Format canonique des documents

Chaque fichier `.md` suit UNE convention unifiée (sortie de `extraction/normalizer.py`) :

```markdown
---
source: chapter_convolutional-neural-networks_channels
source_type: pdf          # pdf | web
source_url: https://...   # optionnel, web uniquement
---

<!-- loc page=5 -->

## 2.3. Canaux multiples

Texte de la section...

--- [FIGURE] ---
Description générée par Qwen3-VL
--- [/FIGURE] ---
```

### 5.3 Pipeline d'ingestion

```
PDF brut ──→ pdf_extractor.py (GLM-OCR) ──→ markdown
  │                                          │
  │                                     figure_describe.py (Qwen3-VL)
  │                                          │
  └──────────────────────────────────────────┼──→ normalizer.py ──→ format unifié
                                             │
Web ────→ web_scraper.py ────────────────────┘
          web_crawler.py
                                                  │
                                                  ▼
                                          ingest.py ──→ chunking ──→ embeddings ──→ ChromaDB
```

---

## 6. Architecture runtime

```
┌────────────────────────────────────────────────────────────┐
│                      CLI (chat.py / tutor.py)              │
├────────────────────────────────────────────────────────────┤
│                      pipeline.py (orchestrateur)           │
│  answer(query, history=None, k=4, mode="hybrid_rerank")   │
├────────────┬───────────┬──────────────┬───────────────────┤
│ QP         │ retriever │ refusal_gate │ generator         │
│ reformule  │ BM25+     │ M1 (reranker)│ system_prompt +   │
│ décompose  │ dense+    │ M2 (verify)  │ contexte → LLM    │
│            │ RRF+CE    │              │                   │
├────────────┴───────────┴──────────────┴───────────────────┤
│                    Ollama (localhost:11434)                │
│  qwen2.5:14b  │  bge-m3  │  qwen3:8b  │  glm-ocr (off)   │
├───────────────────────────────────────────────────────────┤
│  ChromaDB (dbfig_pc/)  │  BGE-reranker (HuggingFace)     │
└───────────────────────────────────────────────────────────┘
```

---

## 7. Modes opératoires

### 7.1 Premier lancement (setup complet)

```bash
# 1. Cloner le dépôt
cd rag-tutor-v2

# 2. Installer les dépendances
pip install -e .

# 3. Vérifier qu'Ollama tourne
ollama list

# 4. Télécharger les modèles
ollama pull qwen2.5:14b
ollama pull qwen3:8b
ollama pull bge-m3

# 5. Ingérer le corpus (30-60 min)
python -m rag_tutor.ingestion.ingest data/processed/

# 6. Lancer le chat
python -m rag_tutor.cli.chat
```

### 7.2 Extraction PDF (optionnel, si nouveau PDF à ajouter)

```bash
python -m rag_tutor.extraction.pdf_extractor data/raw/pdf/cours.pdf \
    --config config/glmocr_config.yaml \
    --out data/processed/
```

### 7.3 Évaluation

```bash
python -m rag_tutor.evaluation.evaluate --retrieval-mode hybrid_rerank
```

### 7.4 Calibration du seuil de refus

```bash
python -m rag_tutor.evaluation.calibrate
```

---

## 8. Performances attendues

| Métrique | Valeur (système complet) |
|----------|--------------------------|
| **Temps de réponse** | < 30 secondes (cahier des charges) |
| **Hit@4 (retrieval)** | 0.967 |
| **MRR (retrieval)** | 0.867 |
| **Précision refus** | 1.000 |
| **Rappel refus** | 1.000 |
| **F1 refus** | 1.000 |
| **Fidélité (Ragas)** | ~0.88 (baseline dense Run 1 uniquement) |

---

## 9. Limites connues

1. **Dépendance à Ollama** — le pipeline ne fonctionne pas sans serveur Ollama local (pas de fallback API cloud)
2. **Reranker via HuggingFace** — premier téléchargement nécessite connexion internet (~600 Mo). Passe offline avec `export HF_HUB_OFFLINE=1`
3. **Évaluation génération** uniquement sur le Run 1 (baseline dense) pour des raisons de coût Ragas
4. **OCR GLM** nécessite un GPU avec ≥ 16 GB VRAM dédié (ou CPU, très lent)
5. **Corpus monolingue FR/EN** — les embeddings BGE-M3 sont multilingues, mais le système n'a été testé que sur français + anglais technique
6. **Pas de streaming SSE** — les réponses arrivent en bloc, pas token par token
7. **Pas d'authentification** — le CLI est mono-utilisateur
