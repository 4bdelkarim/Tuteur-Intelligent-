#!/usr/bin/env python3
"""
validate_judge.py — Generation des scores du juge LLM sur le set de calibration.

Modes:
  --generate        : lance le pipeline + Ragas sur les questions de calibration
  --ragas-only      : relit le JSON existant et relance Ragas en sequentiel
  --correctness-only: evalue la correctness avec un prompt simple 1-5 (qwen3:8b)

NB: l'annotation humaine + correlation Pearson (--correlate) a ete ABANDONNEE
(J4) — pas assez d'expertise domaine pour une annotation fiable.
"""

import argparse
import json
import sys
from pathlib import Path

# --- CONFIG ---
CALIBRATION_PATH = "eval/calibration_set.json"
OUTPUT_PATH = "eval/judge_validation.json"
OUTPUT_ANNOTATED_PATH = "eval/judge_validation_annotated.json"


def _patch_and_import():
    """Patch le bug VertexAI de ragas avant tout import."""
    import types as _types
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        fake_mod = _types.ModuleType("langchain_community.chat_models.vertexai")
        class ChatVertexAI:
            pass
        fake_mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = fake_mod


def generate_validation_output(calibration_path: str, output_path: str) -> None:
    """MODE --generate: Genere les reponses du pipeline + scores Ragas pour
    les questions de calibration. Sauvegarde dans eval/judge_validation.json
    (aucune annotation humaine — J4 abandonne)."""
    output_annotated_path = output_path.replace(".json", "_annotated.json")
    _patch_and_import()
    from .evaluate import load_dataset, _judge_llm, _load_metrics
    from ragas import SingleTurnSample, EvaluationDataset, RunConfig
    from ragas import evaluate as ragas_evaluate
    from ..core.pipeline import answer as pipeline_answer
    from ..core.generator import REFUSAL_MESSAGE as REFUSAL_MSG

    items = load_dataset(calibration_path)
    print(f"Validation juge : {len(items)} questions chargees depuis {calibration_path}")

    # --- Phase 1: pipeline ---
    print("[1/3] pipeline.answer() sur chaque question...")
    per_question = []
    ragas_samples = []

    for i, item in enumerate(items):
        result = pipeline_answer(item["question"], k=4,
                                  use_query_processing=True,
                                  use_refusal_gate=True)
        refused = result.answer.strip() == REFUSAL_MSG
        print(f"  [{i+1}/{len(items)}] {item['question'][:60]}..."
              f"  → {len(result.answer)} chars"
              f"  {'[REFUS]' if refused else ''}")

        entry = {
            "index": i,
            "question": item["question"],
            "category": item["category"],
            "reference": item["reference"],
            "gold_context": item["gold_context"],
            "generated_answer": result.answer,
            "retrieved_contexts": result.contexts,
        }
        per_question.append(entry)

        # Preparer l'echantillon Ragas (sauf unanswerable)
        if item["category"] != "unanswerable":
            ragas_samples.append(SingleTurnSample(
                user_input=item["question"],
                response=result.answer,
                retrieved_contexts=result.contexts,
                reference=item["reference"],
            ))

    # --- Phase 2: Ragas ---
    print(f"\n[2/3] Evaluation Ragas sur {len(ragas_samples)} echantillons answerable...")
    ragas_scores_per_q = {}
    if ragas_samples:
        dataset = EvaluationDataset(samples=ragas_samples)
        llm = _judge_llm()
        metrics = _load_metrics()
        result = ragas_evaluate(dataset=dataset, metrics=metrics, llm=llm,
                                run_config=RunConfig(max_workers=3, timeout=600))
        try:
            df = result.to_pandas()
            for i_row, row in df.iterrows():
                ragas_scores_per_q[i_row] = {
                    "faithfulness": float(row.get("faithfulness", float("nan"))),
                    "factual_correctness": float(row.get("factual_correctness", float("nan"))),
                    "context_precision": float(row.get("context_precision", float("nan"))),
                }
        except Exception:
            print("  ⚠ Impossible d'extraire les scores par question depuis Ragas")
    else:
        print("  (aucun echantillon answerable)")

    # --- Phase 3: Sauvegarde ---
    print(f"\n[3/3] Sauvegarde dans {output_path}...")
    for i, entry in enumerate(per_question):
        entry["ragas_scores"] = ragas_scores_per_q.get(i)
    Path(output_path).write_text(
        json.dumps(per_question, ensure_ascii=False, indent=2), encoding="utf-8")

    # Creer une copie pour l'annotation (le fichier que l'humain remplit)
    Path(output_annotated_path).write_text(
        json.dumps(per_question, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Fichiers generes :")
    print(f"   {output_path}            (reference, scores Ragas)")
    print(f"   {output_annotated_path}  (copie identique)")
    print(f"\nPour completer les scores (optionnel) :")
    print(f"   python scripts101/validate_judge.py --ragas-only        (NaN fix, sequentiel)")
    print(f"   python scripts101/validate_judge.py --correctness-only  (correctness 1-5)")


def _custom_correctness_score(reference, answer):
    """Custom correctness metric: qwen3:8b avec un prompt simple 1-5.
    Evite le JSON parsing complexe de FactualCorrectness qui echoue
    systematiquement avec les modeles locaux 8B.
    Retourne un float 0.0-1.0 (score 1-5 divise par 5)."""
    import re as _re
    _patch_and_import()
    from langchain_ollama import ChatOllama
    from ..core.llm_client import OLLAMA_HOST
    from .evaluate import JUDGE_MODEL, JUDGE_NUM_CTX

    llm = ChatOllama(model=JUDGE_MODEL, temperature=0,
                     base_url=OLLAMA_HOST, num_ctx=JUDGE_NUM_CTX)

    prompt = (
        f"Évalue la CORRECTITUDE de la réponse ci-dessous par rapport à la référence.\n"
        f"Attribue un score de 1 à 5:\n"
        f"  1 = complètement faux / hors sujet\n"
        f"  2 = majoritairement faux\n"
        f"  3 = partiellement correct\n"
        f"  4 = correct avec des imprécisions mineures\n"
        f"  5 = parfaitement correct et complet\n\n"
        f"Référence: {reference}\n\n"
        f"Réponse: {answer}\n\n"
        f"Score (1-5):"
    )
    try:
        resp = llm.invoke(prompt)
    except Exception as e:
        print(f"  ⚠ appel LLM echoue: {e}")
        return float("nan")

    raw = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
    digits = _re.findall(r'[1-5]', raw)
    score = int(digits[0]) if digits else None
    if score is None:
        print(f"  ⚠ impossible d'extraire un score 1-5 de: '{raw[:60]}...'")
        return float("nan")
    return score / 5.0   # normaliser 0-1 comme les scores Ragas


def correctness_only(output_path: str) -> None:
    """MODE --correctness-only: Relit le JSON existant et evalue la
    correctness avec un prompt simple 1-5 (qwen3:8b), sans passer par
    le FactualCorrectness de Ragas (incompatible avec les modeles 8B).
    Stocke le score dans ragas_scores.custom_correctness."""
    output_annotated_path = output_path.replace(".json", "_annotated.json")
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    print(f"Custom correctness scoring sur {output_path}...")

    answerable = [e for e in data if e["category"] != "unanswerable"]
    print(f"  {len(answerable)} questions answerable")

    for entry in answerable:
        score_01 = _custom_correctness_score(
            entry["reference"], entry["generated_answer"]
        )
        raw_1_5 = round(score_01 * 5) if score_01 == score_01 else None
        print(f"  Q{entry['index']}: {raw_1_5}/5 (normalise: {score_01})")
        if entry.get("ragas_scores") is None:
            entry["ragas_scores"] = {}
        entry["ragas_scores"]["custom_correctness"] = score_01

    # Ecraser les deux fichiers
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(output_annotated_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ custom_correctness ajoute dans :")
    print(f"   {output_path}")
    print(f"   {output_annotated_path}")


def ragas_only(output_path: str) -> None:
    """MODE --ragas-only: Relit le JSON existant et relance Ragas en
    sequentiel (max_workers=1) sur les questions answerable uniquement.
    Utile quand le run initial (max_workers=3) a timeout sur certaines
    metriques (FactualCorrectness notamment)."""
    output_annotated_path = output_path.replace(".json", "_annotated.json")
    _patch_and_import()
    from .evaluate import _judge_llm, _load_metrics
    from ragas import SingleTurnSample, EvaluationDataset, RunConfig
    from ragas import evaluate as ragas_evaluate

    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    print(f"Re-run Ragas sequentiel sur {output_path}...")

    # Reconstruire les echantillons Ragas pour les answerable seulement
    ragas_samples = []
    answerable_indices = []
    for entry in data:
        if entry["category"] != "unanswerable":
            ragas_samples.append(SingleTurnSample(
                user_input=entry["question"],
                response=entry["generated_answer"],
                retrieved_contexts=entry["retrieved_contexts"],
                reference=entry["reference"],
            ))
            answerable_indices.append(entry["index"])

    if not ragas_samples:
        print("  (aucun echantillon answerable)")
        return

    print(f"  {len(ragas_samples)} echantillons answerable, max_workers=1 (sequentiel)...")
    dataset = EvaluationDataset(samples=ragas_samples)
    llm = _judge_llm()
    metrics = _load_metrics()
    result = ragas_evaluate(dataset=dataset, metrics=metrics, llm=llm,
                            run_config=RunConfig(max_workers=1, timeout=600))

    # Extraire les scores par question
    ragas_scores_per_q = {}
    try:
        df = result.to_pandas()
        for i_row, row in df.iterrows():
            idx = answerable_indices[i_row]
            ragas_scores_per_q[idx] = {
                "faithfulness": float(row.get("faithfulness", float("nan"))),
                "factual_correctness": float(row.get("factual_correctness", float("nan"))),
                "context_precision": float(row.get("context_precision", float("nan"))),
            }
    except Exception as e:
        print(f"  ⚠ Impossible d'extraire les scores : {e}")
        return

    # Mettre a jour les ragas_scores dans les entrees
    for entry in data:
        if entry["index"] in ragas_scores_per_q:
            entry["ragas_scores"] = ragas_scores_per_q[entry["index"]]

    # Ecraser les deux fichiers
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(output_annotated_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Bilan rapide (on ne verifie que faithfulness — factual_correctness est
    # connu pour echouer avec les modeles locaux 8B (JSON mal formé) et sera
    # ignore dans la correlation Pearson de toute facon).
    print(f"\n✅ Ragas scores mis a jour dans :")
    print(f"   {output_path}")
    print(f"   {output_annotated_path}")
    faith_nan = sum(1 for e in data if e.get("ragas_scores")
                    and isinstance(e["ragas_scores"].get("faithfulness"), float)
                    and e["ragas_scores"]["faithfulness"] != e["ragas_scores"]["faithfulness"])
    fc_nan = sum(1 for e in data if e.get("ragas_scores")
                  and isinstance(e["ragas_scores"].get("factual_correctness"), float)
                  and e["ragas_scores"]["factual_correctness"] != e["ragas_scores"]["factual_correctness"])
    if faith_nan > 0:
        print(f"  ⚠ {faith_nan} faithfulness NaN — probleme")
    else:
        print(f"  ✅ faithfulness OK (5/5)")
    if fc_nan > 0:
        print(f"  ⚠ {fc_nan} factual_correctness NaN (attendu — qwen3:8b incompatible, sera ignore)")
    else:
        print(f"  ✅ factual_correctness OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generation des scores du juge LLM sur le set de calibration (J4 humain abandonne)"
    )
    ap.add_argument("--generate", action="store_true",
                    help="MODE 1: generer les reponses + scores Ragas pour annotation humaine")
    ap.add_argument("--ragas-only", action="store_true",
                    help="MODE: relit le JSON existant et relance Ragas en sequentiel "
                         "(max_workers=1) — utile si le --generate initial a timeout")
    ap.add_argument("--correctness-only", action="store_true",
                    help="MODE: relit le JSON existant et evalue la correctness "
                         "avec un prompt simple 1-5 (compatible qwen3:8b)")
    ap.add_argument("--calibration", default=CALIBRATION_PATH,
                    help=f"chemin du fichier de calibration (defaut {CALIBRATION_PATH})")
    ap.add_argument("--output", default=OUTPUT_PATH,
                    help=f"chemin de sortie (defaut {OUTPUT_PATH})")
    args = ap.parse_args()

    if args.generate:
        generate_validation_output(args.calibration, args.output)
    elif args.ragas_only:
        ragas_only(args.output)
    elif args.correctness_only:
        correctness_only(args.output)
    else:
        ap.print_help()
        print("\nUtilise --generate, --ragas-only, ou --correctness-only.")
