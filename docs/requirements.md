# Requirements — Tuteur RAG Pédagogique (`rag-tutor v2.1.0`)

> **Document de référence** pour reproduire l'environnement complet du pipeline.

---

## 1. Résumé exécutif

Le tuteur RAG est un pipeline complet qui ingère des PDFs de cours + pages web, les transforme en markdown via GLM-OCR + Qwen3-VL (figures), les indexe dans ChromaDB, puis expose **deux interfaces** :

| Interface | Fichier | Prompt | Mémoire | Usage |
|-----------|---------|--------|---------|-------|
| **Chat direct** | `chat.py` | `DEFAULT_SYSTEM_PROMPT` (professeur) | Aucune | Évaluation Ragas, Q&A rapide |
| **Tuteur socratique** | `tutor.py` | `TUTOR_SYSTEM_PROMPT` (pédagogue) | `ConversationMemory` | Interface interactive |

### Narration : du professeur au tuteur

Le système a été construit en deux phases :

**Phase 1 — Professeur (v2.0)** : Réponse directe et factuelle à chaque question, sans mémoire. Le pipeline est évalué sur un golden dataset via Ragas (Hit@4, MRR, fidélité, refus). Cette phase a permis de valider le retrieval hybride, le double mécanisme de refus (M1/M2) et la calibration des seuils.

**Phase 2 — Tuteur (v2.1)** : Ajout d'une mémoire conversationnelle et d'un ton socratique. Le tuteur :
- Maintient un **historique multi-tours** avec compression automatique (résumé LLM des tours anciens)
- Adopte une **posture pédagogique** : explication → relance → adaptation au niveau
- **Streame** ses réponses token par token (expérience interactive fluide)
- S'**auto-évalue** en temps réel via `--show-eval` (scores reranker, fidélité juge LLM, overlap lexical)

Le pipeline sous-jacent (`pipeline.answer()`) est **unique et stateless** — la mémoire et le prompt sont injectés en paramètres, garantissant qu'évaluation et usage interactif partagent le même code.

### Capacités techniques

- Retrieval hybride (BM25 + dense + cross-encoder reranker + RRF)
- Mécanisme de refus double (M1 pré-génération par reranker + M2 post-génération par juge LLM)
- Query processing (reformulation + décomposition en sous-questions)
- Conversation memory avec fenêtre glissante + résumé LLM
- Per-question evaluation en temps réel (pipeline + juge + retrieval)
- Streaming token par token (mode tuteur uniquement)

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
| `RECENT_WINDOW` | 6 | `memory.py` | Tours récents gardés intacts avant compression |
| `MAX_SUMMARY_TOKENS` | 500 | `memory.py` | Tokens max pour le résumé (fallback troncature) |
| `HISTORY_MAX_TOKENS` | 4000 | `memory.py` | Tokens max pour l'historique formaté dans le prompt |

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

### 6.1 Schéma global

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI                                      │
│  chat.py (Q&A direct)          tutor.py (conversationnel)        │
│  • pas d'historique            • ConversationMemory              │
│  • DEFAULT_SYSTEM_PROMPT       • TUTOR_SYSTEM_PROMPT              │
│  • pas de streaming            • streaming token/token            │
│                                • --show-eval (éval temps réel)    │
├──────────────────────────────────────────────────────────────────┤
│                    pipeline.py (orchestrateur unique)             │
│  answer(query, history=None, system_prompt=None, stream=False)   │
├────────────┬───────────┬──────────────┬──────────────────────────┤
│ QP         │ retriever │ refusal_gate │ generator                │
│ reformule  │ BM25+     │ M1 (reranker)│ system_prompt +          │
│ décompose  │ dense+    │ M2 (verify)  │ contexte + history → LLM │
│            │ RRF+CE    │              │                          │
├────────────┴───────────┴──────────────┴──────────────────────────┤
│                    Ollama (localhost:11434)                       │
│  qwen2.5:14b  │  bge-m3  │  qwen3:8b  │  glm-ocr (off)          │
├──────────────────────────────────────────────────────────────────┤
│  ChromaDB (dbfig_pc/)  │  BGE-reranker (HuggingFace)            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│               Couche conversation (v2.1)                         │
│                                                                  │
│  conversation/memory.py           conversation/prompts.py        │
│  ┌─────────────────────┐         ┌─────────────────────────┐    │
│  │ ConversationMemory   │         │ TUTOR_SYSTEM_PROMPT     │    │
│  │ • fenêtre glissante  │         │ • posture socratique    │    │
│  │ • résumé LLM         │         │ • expliquer + relancer  │    │
│  │ • détection sujet    │         │ • adaptation niveau     │    │
│  │ • formatage prompt   │         │ • citations obligatoires│    │
│  └─────────────────────┘         └─────────────────────────┘    │
│                                                                  │
│  evaluation/per_question.py                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ evaluate_response() — 3 niveaux par question              │   │
│  │  🔧 Pipeline : scores reranker, nb contextes, refus      │   │
│  │  ⚖️  Juge LLM : fidélité (ANCRÉ / HORS-CONTEXTE)        │   │
│  │  🔍 Retrieval : overlap lexical, diversité sources       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Flux d'une requête conversationnelle

```
1. utilisateur pose une question
        │
2. tutor.py : memory.add_turn("student", question)
        │
3. tutor.py : history = memory.get_formatted_history()
   → [RÉSUMÉ] (si tours anciens compressés)
   → [DERNIERS ÉCHANGES] (6 derniers tours)
        │
4. pipeline.answer(question, history=history, system_prompt=TUTOR_SYSTEM_PROMPT)
        │
5. query_processing : reformule + décompose la question
   (n'utilise PAS l'historique pour la reformulation — v2.1)
        │
6. retriever : BM25 + dense + RRF + reranker → top-k parents
        │
7. refusal_gate M1 : score reranker < seuil ? → REFUSAL_MESSAGE
        │
8. generator : construit le prompt
   ┌──────────────────────────────────────────┐
   │ HISTORIQUE DE LA CONVERSATION :          │
   │   [RÉSUMÉ] L'étudiant a exploré...       │
   │   [DERNIERS ÉCHANGES]                    │
   │   Étudiant : compare la avec le cnn      │
   │                                          │
   │ DOCUMENTS DE COURS :                     │
   │   [DOC 1] ...  [DOC 2] ...               │
   │                                          │
   │ QUESTION : compare la avec le cnn        │
   └──────────────────────────────────────────┘
        │
9. LLM (qwen2.5:14b) génère la réponse (streaming)
        │
10. refusal_gate M2 : verify_answer() → HORS-CONTEXTE ?
    (désactivé en mode streaming)
        │
11. tutor.py : memory.add_turn("tutor", réponse)
    memory.compress(llm_summarize_fn) → résumé LLM si > 6 tours
        │
12. (optionnel) --show-eval : rapport d'évaluation
```

### 6.3 Stratégie de compression de la mémoire

| État | Comportement |
|------|-------------|
| ≤ `recent_window` tours (défaut 6) | Aucune compression — tous les tours sont gardés intacts |
| > `recent_window` tours | Les tours anciens sont résumés par `qwen2.5:14b` (temperature=0) en 2-3 phrases. Les `recent_window` derniers restent intacts. |
| Échec du LLM summarizer | Fallback : troncature simple à `max_summary_tokens` mots (défaut 500) |
| Historique formaté > 4000 tokens | Les tours les plus anciens sont supprimés (minimum 2 préservés) |

```
Avant compression (8 tours, recent_window=6) :
┌──── Tours 1-2 (anciens) ────┐
│ É : c'est quoi une LSTM ?   │  ← résumés par le LLM
│ T : Une LSTM est un type... │
├──── Tours 3-8 (récents) ────┤
│ É : comment elle résout...  │  ← gardés intacts
│ T : Les portes permettent.. │
│ ...                         │
└─────────────────────────────┘

Après compression, get_formatted_history() retourne :

[RÉSUMÉ DE LA CONVERSATION PRÉCÉDENTE]
L'étudiant a exploré les LSTM, leurs portes (oubli, entrée, sortie)
et la différence avec les RNN classiques. Le tuteur a fourni des
explications structurées avec des citations des documents.

[DERNIERS ÉCHANGES]
Étudiant : compare la avec le cnn
Tuteur : Les CNN utilisent des filtres convolutifs...
...
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

# 6. Lancer le chat (mode Q&A direct)
python -m rag_tutor.cli.chat

# 7. Lancer le tuteur (mode conversationnel)
python -m rag_tutor.cli.tutor
```

### 7.2 Chat direct (mode professeur)

```bash
python -m rag_tutor.cli.chat                # mode simple
python -m rag_tutor.cli.chat --k 6          # plus de contextes
python -m rag_tutor.cli.chat --show-sources # afficher les sources
python -m rag_tutor.cli.chat --no-query-processing --no-refusal-gate  # mode dégradé rapide
```

### 7.3 Tuteur conversationnel (mode socratique)

```bash
# Mode conversationnel standard
python -m rag_tutor.cli.tutor

# Avec évaluation temps réel par question
python -m rag_tutor.cli.tutor --show-eval

# Évaluation sans le juge LLM (plus rapide)
python -m rag_tutor.cli.tutor --show-eval --no-judge

# Fenêtre de mémoire réduite (test compression)
python -m rag_tutor.cli.tutor --recent-window 2

# Avec sources et plus de contextes
python -m rag_tutor.cli.tutor --show-sources --k 6
```

**Commandes interactives :**
- `quit` / `exit` / `q` — sortir
- `clear` — réinitialiser la conversation (vide la mémoire)
- `Ctrl+C` / `Ctrl+D` — sortie propre

### 7.4 Flags du tuteur

| Flag | Défaut | Description |
|------|--------|-------------|
| `--k` | 4 | Nombre de contextes récupérés |
| `--show-eval` | off | Affiche le rapport d'évaluation après chaque réponse |
| `--no-judge` | off | Désactive le juge LLM dans `--show-eval` (garde pipeline + retrieval) |
| `--show-sources` | off | Affiche les sources des passages récupérés |
| `--no-query-processing` | off | Saute la reformulation/décomposition (1 appel LLM en moins) |
| `--no-refusal-gate` | off | Désactive le refus explicite M1+M2 |
| `--recent-window` | 6 | Nombre de tours récents gardés intacts avant compression |

### 7.5 Extraction PDF (optionnel, si nouveau PDF à ajouter)

```bash
python -m rag_tutor.extraction.pdf_extractor data/raw/pdf/cours.pdf \
    --config config/glmocr_config.yaml \
    --out data/processed/
```

### 7.6 Évaluation sur golden dataset

```bash
# Évaluation complète (Ragas + métriques custom)
python -m rag_tutor.evaluation.evaluate eval/golden_dataset_v2.json

# Évaluation rapide (sans Ragas, pour itérer)
python -m rag_tutor.evaluation.evaluate eval/golden_dataset_v2.json --no-ragas

# Test sur N questions (vérifier que tout fonctionne)
python -m rag_tutor.evaluation.evaluate eval/golden_dataset_v2.json --limit 5

# Ablation retrieval
python -m rag_tutor.evaluation.evaluate eval/golden_dataset_v2.json --retrieval-mode dense
```

### 7.7 Calibration du seuil de refus

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
6. **Query processing sans historique** (v2.1) — la reformulation/décomposition de la question n'utilise pas l'historique de conversation. Une question comme « compare-la avec le CNN » est reformulée sans savoir que « la » = LSTM. Le LLM comprend le contexte via l'historique injecté dans le prompt de génération, mais le retrieval pourrait être amélioré par une réécriture context-aware.
7. **Pas d'authentification** — le CLI est mono-utilisateur
8. **M2 désactivé en streaming** — la vérification post-génération (`verify_answer`) nécessite la réponse complète et n'est pas exécutée en mode streaming (`tutor.py`). Seul le M1 (reranker) protège contre les réponses hors-contexte en mode conversationnel.
