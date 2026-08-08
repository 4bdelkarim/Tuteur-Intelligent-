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

JUDGE_MODEL = "qwen2.5:3b"   # DOIT differer de GEN_MODEL (llm_client.py, "qwen2.5:7b") -- cf. bug historique Colab.
                              # C'est le seul autre modele pulle sur Colab -- mais un juge PLUS FAIBLE que le
                              # generateur est lui-meme un risque (juge moins capable de detecter les erreurs
                              # fines) : si tu peux pull un 3e modele (ex. qwen2.5:14b, deja documente comme
                              # choix de juge avant), ce serait plus rigoureux que 3b.
K_RETRIEVAL = 4

# Ollama sur l'instance PARTAGEE (11434) ne parallelise pas vraiment (-np 1,
# jamais reconfigure -- confirme). Depuis le passage a l'instance DEDIEE 11500
# (NUM_PARALLEL=3, validee sur GLM-OCR -- ~450 t/s a 2 slots, stable a 3, degrade
# au-dela), max_workers peut suivre cette meme valeur -- remonter au-dela de 3
# reproduirait le meme phenomene de contention deja observe (2 slots ~440t/s
# chacun, 3 slots ~250t/s chacun -- debit total en baisse, pas en hausse).
RAGAS_MAX_WORKERS = 3
RAGAS_TIMEOUT = 600

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

def _is_relevant(gold_context, hit):
    """Un hit (parent recupere) est pertinent si un chunk source de la question
    (gold_context, utilise par DeepEval pour la generer) est contenu dans son
    texte. Heuristique TEXTUELLE -- si ton JSON expose plutot un identifiant
    (page/source/parent_id) qu'un texte brut, remplace par une comparaison
    directe sur cet identifiant : plus fiable qu'un match de texte."""
    return any(gc.strip() and gc.strip() in hit["text"] for gc in gold_context)


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


def is_refusal(answer_text):
    return answer_text.strip() == REFUSAL_MESSAGE


# =====================================================
# 3) EVALUATION RAGAS (faithfulness, context precision...)
# =====================================================

def _judge_llm():
    """base_url explicite (port natif) -- meme raison que le Client explicite
    de embeddings.py/llm_client.py : ChatOllama resoudrait sinon son host via
    OLLAMA_HOST ou un defaut interne, potentiellement pollue par une config
    heritee (tunnel ngrok Colab, instance orpheline sur port aleatoire...)."""
    _patch_ragas_vertexai_bug()
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    from ..core.llm_client import OLLAMA_HOST
    return LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0, base_url=OLLAMA_HOST))


def _load_metrics():
    """Faithfulness + ContextPrecision sont purement LLM-based (pas besoin
    d'embeddings). Import CONFIRME fonctionnel sur ragas 0.4.3 (juste un
    DeprecationWarning -- 'sera supprime en v1.0', mais marche aujourd'hui).
    Le chemin de remplacement ragas.metrics.collections existe deja mais
    demande de passer `llm` a CHAQUE metrique individuellement au lieu du
    `llm=` partage de evaluate() utilise ici -- pas adopte pour l'instant,
    a revisiter si ragas.metrics est vraiment retire dans une v1.0 future."""
    _patch_ragas_vertexai_bug()
    from ragas.metrics import Faithfulness, FactualCorrectness
    try:
        from ragas.metrics import ContextPrecision
    except ImportError:
        from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision
    return [Faithfulness(), ContextPrecision(), FactualCorrectness()]


def run_ragas(samples):
    """`samples` : liste de ragas.SingleTurnSample deja remplis. Si tu ajoutes une
    metrique qui a besoin d'embeddings (ResponseRelevancy, similarite...), il
    faudra aussi passer `embeddings=` a evaluate() (ex. en enveloppant
    embeddings.BGEEmbeddings avec ragas.embeddings.LangchainEmbeddingsWrapper --
    non fait ici car Faithfulness/ContextPrecision/FactualCorrectness n'en ont pas besoin).

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

    skip_ragas=True : saute ENTIEREMENT la phase juge (Faithfulness/ContextPrecision/
    FactualCorrectness) -- de loin la plus lente (plusieurs minutes/question contre
    ~11s pour le pipeline seul). Utile pour iterer vite sur le retrieval/refusal_gate/
    prompts SANS attendre le juge a chaque test. hit@k/MRR/refusal_correctness restent
    calcules (ils viennent du pipeline, pas de Ragas). JAMAIS pour les chiffres finaux
    du rapport -- juste pour l'iteration."""
    _patch_ragas_vertexai_bug()
    from ragas import SingleTurnSample
    import time

    items = load_dataset(dataset_path)
    if limit is not None:
        items = items[:limit]
    samples, hit_scores, mrr_scores, refusal_flags = [], [], [], []
    false_refusals, false_answers = [], []   # detail : refuse a tort / repond a tort (n'a pas refuse alors qu'il aurait du)

    iterator = items
    has_tqdm = False
    if verbose:
        print(f"[1/2] pipeline.answer() sur {len(items)} questions "
              f"(reformulation + retrieval + generation -- generalement la phase la plus longue)...")
        try:
            from tqdm import tqdm
            iterator = tqdm(items, desc="Questions", unit="q")
            has_tqdm = True
        except ImportError:
            print("  (tqdm absent -- un print par question a la place)")
    t0 = time.time()

    for i, item in enumerate(iterator):
        if verbose and not has_tqdm:
            print(f"  [{i + 1}/{len(items)}] {item['question'][:60]}")

        result = pipeline_answer(item["question"], k=k, use_query_processing=use_query_processing,
                                  use_refusal_gate=use_refusal_gate)

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

    if verbose:
        print(f"[1/2] termine en {time.time() - t0:.0f}s.")

    if skip_ragas:
        if verbose:
            print("[2/2] SAUTE (--no-ragas) -- hit@k/MRR/refusal_correctness seulement, "
                  "pas de Faithfulness/ContextPrecision/FactualCorrectness.")
        ragas_scores = {}
    else:
        if verbose:
            print(f"[2/2] evaluation Ragas sur {len(samples)} echantillon(s) "
                  f"(plusieurs appels LLM par metrique -- 3 metriques ici -> patiente)...")
        t1 = time.time()
        ragas_scores = run_ragas(samples)
        if verbose:
            print(f"[2/2] termine en {time.time() - t1:.0f}s.")

    report = {
        "n_questions": len(items),
        "n_evalues_ragas": len(samples) if not skip_ragas else 0,
        f"hit@{k}": (sum(hit_scores) / len(hit_scores)) if hit_scores else None,
        "mrr": (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else None,
        "refusal_correctness": (sum(refusal_flags) / len(refusal_flags)) if refusal_flags else None,
        "faux_refus": len(false_refusals),          # a refuse alors que la question etait repondable
        "faux_non_refus": len(false_answers),        # a repondu (invente ?) alors qu'il fallait refuser
        **ragas_scores,
    }
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
             history_path="eval_history.jsonl"):
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
    ap.add_argument("--no-ragas", action="store_true",
                    help="saute ENTIEREMENT la phase juge (Faithfulness/ContextPrecision/FactualCorrectness) -- "
                         "de loin la plus lente. hit@k/MRR/refusal_correctness restent calcules. "
                         "Pour iterer vite sur retrieval/refusal_gate/prompts -- JAMAIS pour les chiffres finaux.")
    ap.add_argument("--no-save", action="store_true", help="ne pas enregistrer ce run dans l'historique JSONL")
    ap.add_argument("--history", default="eval_history.jsonl", help="chemin du fichier d'historique (defaut eval_history.jsonl)")
    args = ap.parse_args()

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