#!/usr/bin/env python3
"""
evaluate_rag.py — SEULE RESPONSABILITE : evaluer pipeline.py (le VRAI pipeline,
pas une copie) sur un golden dataset, via Ragas pour les metriques LLM-jugees
(faithfulness, context precision...) + des metriques custom que Ragas ne
fournit pas nativement : Hit@k, MRR (retrieval), refusal correctness (refus).

Ne duplique NI la logique de retrieval NI celle de generation : chaque question
du dataset passe par pipeline.answer(), exactement comme en usage reel -- c'est
tout l'objectif de cette refonte depuis le debut.

ATTENTION SCHEMA JSON (a verifier avant le premier run reel) : load_dataset()
essaie plusieurs noms de champs courants en sortie DeepEval Synthesizer, mais
N'A PAS ete verifie contre TON fichier reel -- regarde le docstring de
load_dataset() et ajuste si les noms different (ou montre-moi un extrait).

ATTENTION JUDGE_MODEL : doit rester DIFFERENT de llm_client.GEN_MODEL -- c'est
exactement le bug JUDGE_MODEL/GEN_MODEL confondus deja rencontre sur Colab.
Un avertissement se declenche automatiquement si les deux coincident.

ATTENTION RAGAS : bug CONFIRME dans ragas 0.4.x (issue GitHub vibrantlabsai/
ragas#2745/#2741) -- ragas/llms/base.py importe encore ChatVertexAI depuis
l'ancien emplacement langchain_community.chat_models.vertexai, supprime dans
les versions recentes de langchain-community (deplace vers le package separe
langchain-google-vertexai, jamais installe ni utilise ici puisqu'on tourne sur
Ollama). _patch_ragas_vertexai_bug() ci-dessous neutralise cet import cassé
AVANT d'importer ragas -- sans ca, `import ragas` plante meme si on n'utilise
jamais Vertex AI. Alternative si ce patch ne suffit pas : pip install
"ragas==0.3.9" (version d'avant ce bug, confirmee fonctionnelle sur l'issue).

Dependances (en plus de celles du pipeline) :
  pip install ragas langchain-ollama
Prerequis : la base doit deja etre indexee (ingest.py) avant de lancer ce script.
"""

import json
import math
import random
import re
import sys
import types
import warnings
from pathlib import Path

from ..core.pipeline import answer as pipeline_answer
from ..core.generator import REFUSAL_MESSAGE
from ..core.llm_client import GEN_MODEL as _GEN_MODEL


def _patch_ragas_vertexai_bug():
    """Contourne un bug confirme de ragas 0.4.x : `ragas/llms/base.py` importe
    ChatVertexAI (et VertexAI) depuis un emplacement de langchain_community
    supprime depuis (cf. issue GitHub vibrantlabsai/ragas#2745). On n'utilise
    jamais Vertex AI (Ollama uniquement) -- de simples classes factices dans
    sys.modules suffisent a satisfaire l'import sans rien casser d'autre.
    Ne fait rien si l'import fonctionne deja normalement (ragas corrige un
    jour, ou langchain_community qui a garde l'ancien chemin)."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return   # deja importable normalement -- rien a patcher
    except ModuleNotFoundError:
        pass

    fake_mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:   # jamais instanciee reellement -- juste pour satisfaire l'import
        pass

    fake_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = fake_mod

    try:
        import langchain_community.llms as _lcc_llms
        if not hasattr(_lcc_llms, "VertexAI"):
            class VertexAI:
                pass
            _lcc_llms.VertexAI = VertexAI
    except ModuleNotFoundError:
        pass

JUDGE_MODEL = "qwen3:8b"   # DOIT differer de GEN_MODEL (llm_client.py, "qwen2.5:14b") -- cf. bug historique Colab.
                              # Modele texte pur 5.2 GB, bien adapte au role de juge (pas d'overhead vision).
                              # A valider avec J4 (Pearson) avant les runs finaux.
                              # NE PAS revenir a qwen2.5:3b (confirme non fiable a plusieurs reprises).
JUDGE_NUM_CTX = 8192         # Contexte juge reduit a 8192 : le juge ne recoit que question + reponse +
                              # reference + contexte (quelques centaines de tokens). Economise du KV cache
                              # vs le defaut modele (~32K) — impact sur la qualite des jugements a
                              # confirmer avec J4 (Pearson).
K_RETRIEVAL = 4

# Ollama sur l'instance PARTAGEE (11434) ne parallelise pas vraiment (-np 1,
# jamais reconfigure -- confirme). Depuis le passage a l'instance DEDIEE 11500
# (NUM_PARALLEL=3, validee sur GLM-OCR -- ~450 t/s a 2 slots, stable a 3, degrade
# au-dela), max_workers peut suivre cette meme valeur -- remonter au-dela de 3
# reproduirait le meme phenomene de contention deja observe (2 slots ~440t/s
# chacun, 3 slots ~250t/s chacun -- debit total en baisse, pas en hausse).
RAGAS_MAX_WORKERS = 3
RAGAS_TIMEOUT = 600

# Par defaut on reste sequentiel — le parallelisme Python seul (ThreadPoolExecutor)
# n'aide que si Ollama a NUM_PARALLEL > 1. Une fois NUM_PARALLEL configure, passer
# PIPELINE_MAX_WORKERS a 4 (ou =NUM_PARALLEL) pour paralleliser les appels pipeline.
PIPELINE_MAX_WORKERS = 1

if JUDGE_MODEL == _GEN_MODEL:
    warnings.warn(
        f"JUDGE_MODEL ({JUDGE_MODEL}) == GEN_MODEL de llm_client.py ({_GEN_MODEL}) : "
        "risque de biais de juge deja rencontre sur Colab. Verifie que c'est volontaire."
    )


# =====================================================
# 1) CHARGEMENT DU GOLDEN DATASET (DeepEval Synthesizer, JSON)
# =====================================================

def load_dataset(path):
    """Charge le golden dataset et le normalise. SCHEMA CONFIRME (extrait reel
    fourni) :
      question     <- item["question"]
      reference    <- item["ground_truth"]  (pour "unanswerable", c'est une phrase
                       fixe type "L'information n'est pas presente dans le contexte." --
                       pas une reponse factuelle, d'ou l'exclusion des metriques Ragas
                       plus bas pour cette categorie)
      gold_context <- item["contexts"]  (liste des chunks source utilises par DeepEval
                       pour generer la question -- verite terrain retrieval)
      category     <- item["category"]  (single_passage / multi_passage / unanswerable)
    Garde aussi quelques noms alternatifs (input/expected_output/context/
    retrieval_context/additional_metadata.category) au cas ou une autre export
    DeepEval serait utilisee plus tard. Accepte soit une liste JSON directe,
    soit un objet avec une cle "goldens"/"data".
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else (raw.get("goldens") or raw.get("data") or [])

    dataset = []
    for item in items:
        meta = item.get("additional_metadata") or {}
        dataset.append({
            "question": item.get("question") or item.get("input"),
            "reference": item.get("ground_truth") or item.get("expected_output") or item.get("answer") or "",
            "gold_context": item.get("contexts") or item.get("context") or item.get("retrieval_context") or [],
            "category": item.get("category") or meta.get("category"),
        })
    return dataset


# =====================================================
# 2) METRIQUES CUSTOM (ce que Ragas ne fournit pas)
# =====================================================

def _is_relevant(gold_context, hit, overlap_threshold=0.4):
    """Un hit est pertinent si une PART SUBSTANTIELLE des mots distinctifs du
    gold_context se retrouve dans le texte du hit -- PAS une correspondance
    exacte (gc.strip() in hit["text"]), qui echoue des que le chunking source
    du gold_context (DeepEval, chunk_size=800 dans generate_golden_dataset.py)
    differe du chunking d'indexation (chunk_parent_child.py) -- meme quand le
    retrieval trouve objectivement le bon passage.

    overlap_threshold=0.4 (abaisse de 0.5) : fraction des mots (>=4 lettres)
    du gold_context devant se retrouver dans le hit. Depuis l'introduction
    du query processing avec sous-questions, un hit pertinent pour une
    sous-question peut etre plus court que le gold_context complet (question
    composite) -> overlap bidirectionnel (gold->hit ET hit->gold) pour ne
    pas penaliser ces hits partiels mais legitimes."""
    hit_words = set(re.findall(r"\w{4,}", hit["text"].lower()))
    if not hit_words:
        return False
    for gc in gold_context:
        if not gc.strip():
            continue
        gc_words = set(re.findall(r"\w{4,}", gc.lower()))
        if not gc_words:
            continue
        # Bidirectionnel : soit le hit couvre le gold, soit le gold couvre le hit
        overlap_gc = len(gc_words & hit_words) / len(gc_words)
        overlap_hit = len(gc_words & hit_words) / len(hit_words)
        if max(overlap_gc, overlap_hit) >= overlap_threshold:
            return True
    return False


def hit_at_k(gold_context, hits, k):
    if not gold_context:
        return None
    return float(any(_is_relevant(gold_context, h) for h in hits[:k]))


def reciprocal_rank(gold_context, hits):
    if not gold_context:
        return None
    for rank, h in enumerate(hits, start=1):
        if _is_relevant(gold_context, h):
            return 1.0 / rank
    return 0.0


def _relevance_scores(gold_context, hits, k):
    """Calcule un score de pertinence gradue (0-1) pour chaque hit du top-k.
    Utilise le ratio de chevauchement lexical (mots >= 4 lettres) entre le hit
    et chaque gold_context — prend le max (si un hit couvre bien l'un des golds,
    il est pertinent). Permet un nDCG gradue plutot que binaire.
    Bidirectionnel (comme _is_relevant) : max(gold->hit, hit->gold)."""
    if not gold_context or not hits:
        return []
    rels = []
    for h in hits[:k]:
        hit_words = set(re.findall(r"\w{4,}", h["text"].lower()))
        if not hit_words:
            rels.append(0.0)
            continue
        best = 0.0
        for gc in gold_context:
            if not gc.strip():
                continue
            gc_words = set(re.findall(r"\w{4,}", gc.lower()))
            if not gc_words:
                continue
            overlap_gc = len(gc_words & hit_words) / len(gc_words) if gc_words else 0.0
            overlap_hit = len(gc_words & hit_words) / len(hit_words) if hit_words else 0.0
            overlap = max(overlap_gc, overlap_hit)
            if overlap > best:
                best = overlap
        rels.append(best)
    return rels


def precision_at_k(gold_context, hits, k):
    """Precision@k : proportion de hits dans le top-k juges pertinents
    (overlap >= 0.5 avec au moins un gold_context)."""
    if not gold_context or not hits:
        return None
    relevant = sum(1 for h in hits[:k] if _is_relevant(gold_context, h))
    return relevant / min(k, len(hits[:k]))


def ndcg_at_k(gold_context, hits, k):
    """nDCG@k : Discounted Cumulative Gain normalise, utilisant le score de
    pertinence gradue (_relevance_scores) plutot que binaire."""
    if not gold_context or not hits:
        return None
    rels = _relevance_scores(gold_context, hits, k)
    if not rels:
        return None
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95, seed=42):
    """Intervalle de confiance bootstrap (percentile) a `ci`% sur une liste
    de scores. Retourne (lower, upper, mean)."""
    if not scores:
        return None, None, None
    if len(scores) < 2:
        m = sum(scores) / len(scores)
        return m, m, m
    rng = random.Random(seed)
    means = []
    n = len(scores)
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lower = means[int(alpha * n_bootstrap)]
    upper = means[int((1 - alpha) * n_bootstrap)]
    mean = sum(scores) / n
    return lower, upper, mean


def is_refusal(answer_text):
    """Detection de refus : egalite stricte avec le message pipeline (M2/M3)
    OU detection de refus naturels du LLM (phrases courtes indiquant l'incapacite
    a repondre a partir des documents).

    Ne capture PAS les refus suivis d'une reponse factuelle : si le LLM dit
    « je ne peux pas repondre avec les documents, mais [reponse] », c'est
    un faux non-refus legitime que M3 (verify_answer) doit attraper."""
    text = answer_text.strip()
    # Cas 1 : refus explicite du pipeline (M2/M3)
    if text == REFUSAL_MESSAGE:
        return True
    # Cas 2 : refus naturel du LLM — phrase courte sans contenu factuel
    if len(text) < 500:
        lower = text.lower()
        refusal_patterns = [
            "ne permettent pas de répondre",
            "ne peux pas répondre",
            "ne peut pas répondre",
            "ne pouvons pas répondre",
            "ne suis pas en mesure",
            "n'est pas présente",
            "n'est pas présent",
            "n'est pas directement",
            "pas directement liée",
            "pas directement lié",
            "pas dans le contexte",
            "pas dans le corpus",
            "pas dans les documents",
            "ne correspond pas",
            "ne traite pas de",
            "ne traite pas des",
            "ne contient pas",
            "aucun document",
            "documents fournis ne",
            "je ne dispose pas",
            "hors du champ",
            "ne fait pas partie",
            "information non disponible",
            "pas d'information",
            "ne semble pas",
        ]
        if any(pattern in lower for pattern in refusal_patterns):
            return True
    return False


def _bootstrap_all(report_dict, score_lists, n_bootstrap=1000, ci=0.95):
    """Ajoute les IC bootstrap a un dictionnaire de rapport.
    score_lists : {nom_metrique: [scores]} — les scores deja calcules
    par question. Ajoute {nom_metrique}_ci95_lower, _upper, _mean."""
    for name, scores in score_lists.items():
        if not scores:
            continue
        lower, upper, mean = bootstrap_ci(scores, n_bootstrap, ci)
        if lower is not None:
            report_dict[f"{name}_ci95_lower"] = round(lower, 4)
            report_dict[f"{name}_ci95_upper"] = round(upper, 4)


# =====================================================
# 3a) METRIQUE CUSTOM CORRECTNESS (juge LLM 1-5, sans Ragas)
# =====================================================

def _custom_correctness_score(reference, answer, llm=None):
    """Custom correctness: qwen3:8b avec un prompt simple 1-5.
    Evite le JSON parsing complexe de FactualCorrectness (Ragas) qui
    echoue systematiquement avec les modeles locaux 8B.
    Retourne un float 0.0-1.0 (score 1-5 divise par 5), ou NaN si echec.
    Si llm=None, cree un nouveau ChatOllama (usage one-shot)."""
    if llm is None:
        from langchain_ollama import ChatOllama
        from ..core.llm_client import OLLAMA_HOST
        llm = ChatOllama(model=JUDGE_MODEL, temperature=0,
                         base_url=OLLAMA_HOST, num_ctx=JUDGE_NUM_CTX)

    prompt = (
        "Évalue la CORRECTITUDE de la réponse ci-dessous par rapport à la référence.\n"
        "Attribue un score de 1 à 5:\n"
        "  1 = complètement faux / hors sujet\n"
        "  2 = majoritairement faux\n"
        "  3 = partiellement correct\n"
        "  4 = correct avec des imprécisions mineures\n"
        "  5 = parfaitement correct et complet\n\n"
        f"Référence: {reference}\n\n"
        f"Réponse: {answer}\n\n"
        "Score (1-5):"
    )
    try:
        resp = llm.invoke(prompt)
    except Exception:
        return float("nan")

    raw = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
    digits = re.findall(r'[1-5]', raw)
    score = int(digits[0]) if digits else None
    if score is None:
        return float("nan")
    return score / 5.0   # normaliser 0-1 comme les scores Ragas


# =====================================================
# 3) EVALUATION RAGAS (faithfulness, context precision...)
# =====================================================

def _judge_llm():
    """base_url explicite (port natif) -- meme raison que le Client explicite
    de embeddings.py/llm_client.py : ChatOllama resoudrait sinon son host via
    OLLAMA_HOST ou un defaut interne, potentiellement pollue par une config
    heritee (tunnel ngrok Colab, instance orpheline sur port aleatoire...).

    num_ctx reduit a JUDGE_NUM_CTX (8192) au lieu du defaut modele (262144) —
    le juge ne recoit que la question + reponse + contexte + reference, quelques
    centaines de tokens, donc 262K est du gaspillage pur de VRAM."""
    _patch_ragas_vertexai_bug()
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    from ..core.llm_client import OLLAMA_HOST
    return LangchainLLMWrapper(ChatOllama(
        model=JUDGE_MODEL, temperature=0, base_url=OLLAMA_HOST,
        num_ctx=JUDGE_NUM_CTX,
    ))


def _load_metrics():
    """Faithfulness + ContextPrecision sont purement LLM-based (pas besoin
    d'embeddings). Import CONFIRME fonctionnel sur ragas 0.4.3 (juste un
    DeprecationWarning -- 'sera supprime en v1.0', mais marche aujourd'hui).
    Le chemin de remplacement ragas.metrics.collections existe deja mais
    demande de passer `llm` a CHAQUE metrique individuellement au lieu du
    `llm=` partage de evaluate() utilise ici -- pas adopte pour l'instant,
    a revisiter si ragas.metrics est vraiment retire dans une v1.0 future."""
    _patch_ragas_vertexai_bug()
    from ragas.metrics import Faithfulness
    try:
        from ragas.metrics import ContextPrecision
    except ImportError:
        from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision
    # FactualCorrectness (Ragas) VOLONTAIREMENT EXCLU : son parsing JSON complexe
    # echoue systematiquement avec les modeles locaux 8B (cf. docstring de
    # _custom_correctness_score) -- remplace par la metrique custom 1-5
    # (custom_correctness), calculee en phase [2/3], hors Ragas.
    return [Faithfulness(), ContextPrecision()]


def run_ragas(samples):
    """`samples` : liste de ragas.SingleTurnSample deja remplis. Si tu ajoutes une
    metrique qui a besoin d'embeddings (ResponseRelevancy, similarite...), il
    faudra aussi passer `embeddings=` a evaluate() (ex. en enveloppant
    embeddings.BGEEmbeddings avec ragas.embeddings.LangchainEmbeddingsWrapper --
    non fait ici car Faithfulness/ContextPrecision n'en ont pas besoin ; la
    correctness est calculee separement (custom_correctness, juge LLM 1-5).

    RunConfig ADAPTE a un Ollama LOCAL : le defaut Ragas est max_workers=16 /
    timeout=180s, concu pour une API distante qui parallelise vraiment (OpenAI...).
    Un Ollama local ne traite reellement qu'1-2 requetes a la fois (meme GPU,
    meme modele charge) -- 16 jobs concurrents saturent la file d'attente et la
    plupart timeout avant meme d'etre traites (symptome : presque tous les jobs
    en TimeoutError sauf 1-2, et le seul qui reussit prend ~= au timeout par
    defaut). RAGAS_MAX_WORKERS/RAGAS_TIMEOUT ci-dessous sont des constantes
    modifiables si jamais OLLAMA_NUM_PARALLEL est configure plus haut."""
    if not samples:
        return {}
    _patch_ragas_vertexai_bug()
    from ragas import evaluate, EvaluationDataset, RunConfig

    dataset = EvaluationDataset(samples=samples)
    run_config = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=RAGAS_TIMEOUT)
    result = evaluate(dataset=dataset, metrics=_load_metrics(), llm=_judge_llm(), run_config=run_config)
    try:
        return dict(result)
    except Exception:
        pass
    try:
        # repli : certains echantillons en echec (OUTPUT_PARSING_FAILURE, timeout...)
        # peuvent empecher dict(result) -- to_pandas() degrade mieux, moyenne en
        # ignorant les NaN. Reste un dict de floats, JSON-serialisable.
        means = result.to_pandas().mean(numeric_only=True).to_dict()
        warnings.warn("dict(result) a echoue (probablement des echantillons en erreur) -- "
                       "scores calcules via to_pandas().mean(), NaN ignores.")
        return means
    except Exception:
        # dernier recours : JAMAIS l'objet EvaluationResult brut (pas JSON-serialisable,
        # ferait planter save_run()) -- une string est toujours sure a serialiser.
        warnings.warn("Conversion de l'EvaluationResult Ragas impossible (dict et to_pandas "
                       "ont echoue) -- rapport degrade a une representation texte brute.")
        return {"_raw_result_str": str(result)}


# =====================================================
# 4) ORCHESTRATION : dataset -> pipeline.answer() -> metriques
# =====================================================

def run_evaluation(dataset_path, k=K_RETRIEVAL, verbose=True, limit=None, use_query_processing=True,
                    use_refusal_gate=True, skip_ragas=False):
    """verbose=True (defaut) : affiche la progression question par question et
    le temps par phase. AUCUNE sortie avant ca ne veut PAS dire que ca bloque --
    pipeline.answer() fait 2-3 appels LLM par question (reformulation, retrieval,
    generation) et run_ragas() en refait plusieurs par metrique/echantillon
    ENSUITE : sur un Ollama local (Colab, GPU partage), un run de quelques
    dizaines de questions se compte raisonnablement en dizaines de minutes,
    pas en secondes -- et bien plus si Ollama tourne sur CPU (pas de GPU
    disponible) : compte alors en heures pour un dataset complet, pas en minutes.

    limit=N : ne traite que les N premieres questions -- utile pour mesurer le
    temps reel par question (et decider si un run complet est realiste) avant
    de lancer tout le dataset, surtout sans GPU.

    use_query_processing=False : saute la reformulation/decomposition (1 appel
    LLM de moins par question) -- mode degrade pour CPU pur, pas pour les
    chiffres finaux du rapport.

    skip_ragas=True : saute ENTIEREMENT la phase juge (Faithfulness/ContextPrecision
    + custom_correctness) -- de loin la plus lente. hit@k/MRR/refusal_correctness restent
    calcules (ils viennent du pipeline, pas de Ragas). JAMAIS pour les chiffres finaux
    du rapport -- juste pour iterer vite sur retrieval/refusal_gate/prompts."""
    _patch_ragas_vertexai_bug()
    from ragas import SingleTurnSample
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    items = load_dataset(dataset_path)
    if limit is not None:
        items = items[:limit]
    samples, hit_scores, mrr_scores, precision_scores, ndcg_scores, refusal_flags = [], [], [], [], [], []
    correct_scores = []   # custom correctness par question (0-1)
    false_refusals, false_answers = [], []   # detail : refuse a tort / repond a tort (n'a pas refuse alors qu'il aurait du)
    # Scores par categorie pour le rapport detaille (J3)
    by_cat = {}  # {category: {hit: [], mrr: [], precision: [], ndcg: [], correct: [], refusal_correct: []}}

    n_items = len(items)
    has_tqdm = False
    if verbose:
        print(f"[1/3] pipeline.answer() sur {n_items} questions "
              f"(reformulation + retrieval + generation -- generalement la phase la plus longue)...")
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            print("  (tqdm absent -- un print par question a la place)")
    t0 = time.time()

    def _process_one(idx_item):
        i, item = idx_item
        result = pipeline_answer(item["question"], k=k, use_query_processing=use_query_processing,
                                  use_refusal_gate=use_refusal_gate)
        return i, item, result

    if PIPELINE_MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=PIPELINE_MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, (i, item)): i for i, item in enumerate(items)}
            results_by_idx = {}
            for future in (tqdm(as_completed(futures), total=n_items, desc="Questions", unit="q")
                           if has_tqdm else as_completed(futures)):
                try:
                    i, item, result = future.result()
                    results_by_idx[i] = (item, result)
                except Exception as e:
                    print(f"\n  [ERREUR] question en echec : {e}")
            # Re-order, skip failed entries
            ordered = [results_by_idx[i] for i in range(n_items) if i in results_by_idx]
    else:
        iterator = enumerate(items)
        if has_tqdm:
            iterator = tqdm(list(iterator), desc="Questions", unit="q")
        ordered = []
        for i, item in iterator:
            if verbose and not has_tqdm:
                print(f"  [{i + 1}/{n_items}] {item['question'][:60]}")
            _, item, result = _process_one((i, item))
            ordered.append((item, result))

    for item, result in ordered:
        if item["category"] is not None:
            expected_refusal = (item["category"] == "unanswerable")
            actual_refusal = is_refusal(result.answer)
            refusal_flags.append(actual_refusal == expected_refusal)
            if actual_refusal and not expected_refusal:
                false_refusals.append(item["question"])
            elif not actual_refusal and expected_refusal:
                false_answers.append(item["question"])

        if item["category"] == "unanswerable":
            continue   # pas de reference factuelle a comparer -> exclu des metriques Ragas / Hit@k / MRR

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result.answer,
            retrieved_contexts=result.contexts,
            reference=item["reference"],
        ))

        h = hit_at_k(item["gold_context"], result.hits, k)
        if h is not None:
            hit_scores.append(h)
        m = reciprocal_rank(item["gold_context"], result.hits)
        if m is not None:
            mrr_scores.append(m)
        p = precision_at_k(item["gold_context"], result.hits, k)
        if p is not None:
            precision_scores.append(p)
        n = ndcg_at_k(item["gold_context"], result.hits, k)
        if n is not None:
            ndcg_scores.append(n)

        # Accumuler les scores par categorie
        cat = item["category"] or "unknown"
        if cat not in by_cat:
            by_cat[cat] = {"hit": [], "mrr": [], "precision": [], "ndcg": [], "correct": [], "refusal_correct": []}
        if h is not None:
            by_cat[cat]["hit"].append(h)
        if m is not None:
            by_cat[cat]["mrr"].append(m)
        if p is not None:
            by_cat[cat]["precision"].append(p)
        if n is not None:
            by_cat[cat]["ndcg"].append(n)

    if verbose:
        print(f"[1/3] termine en {time.time() - t0:.0f}s.")

    # --- Phase 2: Custom correctness (toujours, sauf si --no-ragas) ---
    if skip_ragas:
        if verbose:
            print("[2/3] custom correctness SAUTEE (--no-ragas).")
    else:
        if verbose:
            print(f"[2/3] custom correctness sur les reponses answerable...")
        t1 = time.time()
        from langchain_ollama import ChatOllama
        from ..core.llm_client import OLLAMA_HOST
        cc_llm = ChatOllama(model=JUDGE_MODEL, temperature=0,
                            base_url=OLLAMA_HOST, num_ctx=JUDGE_NUM_CTX)
        answerable_items = [(item, result) for item, result in ordered
                            if item["category"] != "unanswerable"]
        for item, result in answerable_items:
            cc = _custom_correctness_score(item["reference"], result.answer, llm=cc_llm)
            if cc == cc:   # pas NaN
                correct_scores.append(cc)
                cat = item["category"] or "unknown"
                if cat not in by_cat:
                    by_cat[cat] = {"hit": [], "mrr": [], "precision": [], "ndcg": [], "correct": [], "refusal_correct": []}
                elif "correct" not in by_cat[cat]:
                    by_cat[cat]["correct"] = []
                by_cat[cat]["correct"].append(cc)
        if verbose:
            n_ok = len(correct_scores)
            n_total = len(answerable_items)
            print(f"[2/3] custom correctness termine en {time.time() - t1:.0f}s "
                  f"({n_ok}/{n_total} scores OK)")

    # --- Phase 3: Ragas (faithfulness, context_precision) — sautee avec --no-ragas ---
    if skip_ragas:
        if verbose:
            print("[3/3] Ragas SAUTE (--no-ragas).")
        ragas_scores = {}
    else:
        if verbose:
            print(f"[3/3] evaluation Ragas sur {len(samples)} echantillon(s)...")
        t2 = time.time()
        ragas_scores = run_ragas(samples)
        if verbose:
            print(f"[3/3] termine en {time.time() - t2:.0f}s.")

    # --- metriques refusal (P/R/F1) ---
    # Derivees des compteurs deja collectes dans la boucle principale (pas de 2e passe)
    # Note: on compte depuis `ordered` (pas `items`) pour rester coherent en mode parallele
    # ou des questions peuvent echouer et ne pas apparaitre dans `ordered`.
    fp = len(false_refusals)
    fn = len(false_answers)
    n_unanswerable = sum(1 for item, _ in ordered if item["category"] == "unanswerable")
    n_answerable = sum(1 for item, _ in ordered if item["category"] != "unanswerable")
    tp = n_unanswerable - fn    # refuse correctement
    tn = n_answerable - fp      # repond correctement
    refusal_precision_val = tp / (tp + fp) if (tp + fp) > 0 else None
    refusal_recall_val = tp / (tp + fn) if (tp + fn) > 0 else None
    refusal_f1_val = (2 * refusal_precision_val * refusal_recall_val / (refusal_precision_val + refusal_recall_val)
                      if (refusal_precision_val is not None and refusal_recall_val is not None
                          and (refusal_precision_val + refusal_recall_val) > 0)
                      else None)

    report = {
        "n_questions": len(items),
        "n_evalues_ragas": len(samples) if not skip_ragas else 0,
        f"hit@{k}": (sum(hit_scores) / len(hit_scores)) if hit_scores else None,
        "mrr": (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else None,
        f"precision@{k}": (sum(precision_scores) / len(precision_scores)) if precision_scores else None,
        f"ndcg@{k}": (sum(ndcg_scores) / len(ndcg_scores)) if ndcg_scores else None,
        "refusal_correctness": (sum(refusal_flags) / len(refusal_flags)) if refusal_flags else None,
        "refusal_precision": refusal_precision_val,
        "refusal_recall": refusal_recall_val,
        "refusal_f1": refusal_f1_val,
        "faux_refus": len(false_refusals),
        "faux_non_refus": len(false_answers),
        "custom_correctness": (sum(correct_scores) / len(correct_scores)) if correct_scores else None,
        **ragas_scores,
    }

    # --- bootstrap IC 95% sur les metriques par question ---
    _bootstrap_all(report, {
        f"hit@{k}": hit_scores,
        "mrr": mrr_scores,
        f"precision@{k}": precision_scores,
        f"ndcg@{k}": ndcg_scores,
        "custom_correctness": correct_scores,
    })

    # --- rapport par categorie (J3) ---
    report["by_category"] = {}
    for cat in sorted(by_cat.keys()):
        scores = by_cat[cat]
        cat_report = {"n": sum(1 for item in items if (item.get("category") or "unknown") == cat)}
        for metric_name in ["hit", "mrr", "precision", "ndcg", "correct"]:
            sc = scores[metric_name]
            if sc:
                avg = sum(sc) / len(sc)
                lower, upper, _mean = bootstrap_ci(sc)
                cat_report[metric_name] = round(avg, 4)
                if lower is not None:
                    cat_report[f"{metric_name}_ci95_lower"] = round(lower, 4)
                    cat_report[f"{metric_name}_ci95_upper"] = round(upper, 4)
            else:
                cat_report[metric_name] = None
        # Refusal correctness pour cette categorie (si applicable)
        cat_refusal_flags = [flag for (item, _result), flag
                             in zip(ordered, refusal_flags)
                             if (item.get("category") or "unknown") == cat]
        if cat_refusal_flags:
            cat_report["refusal_correctness"] = round(sum(cat_refusal_flags) / len(cat_refusal_flags), 4)
        report["by_category"][cat] = cat_report
    if verbose and (false_refusals or false_answers):
        print("\nDetail des erreurs de refus :")
        for q in false_refusals:
            print(f"  [faux refus]      {q[:80]}")
        for q in false_answers:
            print(f"  [faux non-refus]  {q[:80]}")
    return report


# =====================================================
# 5) SAUVEGARDE DES RUNS (historique, pour visualisations ulterieures)
# =====================================================

def save_run(report, dataset_path, k, use_query_processing=True, use_refusal_gate=True,
             retrieval_mode="hybrid_rerank", history_path="eval_history.jsonl"):
    """Enregistre ce run (horodatage + config + rapport) en AJOUT dans un fichier
    JSONL -- une ligne JSON par run, jamais un unique tableau JSON reecrit en
    entier : append-only, aucun risque de corrompre tout l'historique si
    interrompu en cours d'ecriture (utile pour un run de plusieurs heures).
    Chaque ligne est un objet JSON independant -- cf. load_history() pour
    recharger l'historique complet en vue d'une visualisation."""
    import datetime
    from ..core.llm_client import GEN_MODEL, MAX_TOKENS

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "config": {
            "gen_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "max_tokens": MAX_TOKENS,
            "k": k,
            "use_query_processing": use_query_processing,
            "use_refusal_gate": use_refusal_gate,
            "retrieval_mode": retrieval_mode,
        },
        "report": report,
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def load_history(history_path="eval_history.jsonl"):
    """Recharge tout l'historique des runs sauvegardes (liste de dicts, meme
    forme que les records ecrits par save_run()) -- pour construire des
    visualisations comparant plusieurs runs (avant/apres un changement)."""
    try:
        with open(history_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evalue le pipeline RAG complet sur un golden dataset (Ragas + Hit@k/MRR/refusal correctness).")
    ap.add_argument("dataset_path", help="JSON du golden dataset (DeepEval Synthesizer)")
    ap.add_argument("--limit", type=int, default=None,
                    help="ne traite que les N premieres questions -- teste d'abord avec une petite valeur, surtout sans GPU")
    ap.add_argument("--k", type=int, default=K_RETRIEVAL, help=f"nb de parents recuperes par (sous-)question (defaut {K_RETRIEVAL})")
    ap.add_argument("--quiet", action="store_true", help="desactive la barre de progression et les logs de phase")
    ap.add_argument("--no-query-processing", action="store_true",
                    help="saute la reformulation/decomposition (1 appel LLM en moins/question) -- mode degrade CPU, pas pour les chiffres finaux")
    ap.add_argument("--no-refusal-gate", action="store_true",
                    help="desactive la decision de refus explicite (retrouve l'ancien comportement entierement implicite au prompt)")
    ap.add_argument("--retrieval-mode", choices=["dense", "hybrid", "hybrid_rerank"],
                    default="hybrid_rerank",
                    help="mode de retrieval pour l'ablation : dense (baseline) | hybrid | hybrid_rerank (pipeline final)")
    ap.add_argument("--no-ragas", action="store_true",
                    help="saute Faithfulness/ContextPrecision (Ragas) — custom_correctness reste calculee. "
                         "Pour les chiffres finaux sans le cout du parsing Ragas.")
    ap.add_argument("--no-save", action="store_true", help="ne pas enregistrer ce run dans l'historique JSONL")
    ap.add_argument("--history", default="eval_history.jsonl", help="chemin du fichier d'historique (defaut eval_history.jsonl)")
    args = ap.parse_args()

    # Ablation : on bascule le MODE du retriever AVANT tout appel -- retrieve()
    # lit MODE au moment de l'appel (global module, retriever_hybride.py), donc
    # changer l'attribut ici suffit pour tout le run, sans toucher au pipeline.
    from ..core import retriever
    retriever_hybride.MODE = args.retrieval_mode

    report = run_evaluation(args.dataset_path, k=args.k, verbose=not args.quiet, limit=args.limit,
                             use_query_processing=not args.no_query_processing,
                             use_refusal_gate=not args.no_refusal_gate,
                             skip_ragas=args.no_ragas)
    print("=" * 60)
    print("RAPPORT D'EVALUATION")
    print("=" * 60)
    for key, value in report.items():
        print(f"  {key:<22} {value}")

    if not args.no_save:
        save_run(report, args.dataset_path, args.k,
                  use_query_processing=not args.no_query_processing,
                  use_refusal_gate=not args.no_refusal_gate,
                  retrieval_mode=args.retrieval_mode,
                  history_path=args.history)
        print(f"\n(run enregistre dans {args.history})")

# --- SMOKE-TEST rapide avant un run complet (verifie juste que Ragas/Ollama
# repondent, sans passer par tout le dataset) :
#
#   from ragas import SingleTurnSample
#   from evaluate_rag import run_ragas
#   s = SingleTurnSample(user_input="2+2?", response="4",
#                        retrieved_contexts=["2+2 fait 4."], reference="4")
#   print(run_ragas([s]))