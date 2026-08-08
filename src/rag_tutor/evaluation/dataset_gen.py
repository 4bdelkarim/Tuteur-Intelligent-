#!/usr/bin/env python3
"""
generate_golden_dataset.py — genere un golden dataset via DeepEval Synthesizer,
a partir du corpus source deja normalise (.md), avec un souci explicite de
QUALITE des ground_truth ET des comptes EXACTS par categorie (single_passage /
multi_passage / unanswerable -- pas juste des maximums approximatifs).

LE PROBLEME IDENTIFIE SUR LE DATASET PRECEDENT : certains expected_output
depassaient largement le contenu du contexte source (ex. une question sur la
detection de wake words dont le ground_truth mentionnait MFCC/SVM/CNN, absents
du contexte fourni). Un systeme correctement ancre ne peut structurellement pas
matcher ce genre de reference -- ca plombe factual_correctness independamment
de la qualite reelle du systeme.

CE QUI CHANGE ICI :
  - StylingConfig force explicitement expected_output a rester dans les bornes
    strictes du contexte fourni.
  - FiltrationConfig filtre les questions de mauvaise qualite en amont (seuil
    releve a 0.6, defaut deepeval 0.5).
  - qwen2.5:32b (H100 disponible desormais) comme modele de generation/critique
    -- un petit modele respecte moins bien des consignes de grounding strict.
  - Embedder PERSONNALISE pour la construction de contexte : deepeval.models.
    OllamaEmbeddingModel a un bug CONFIRME (poste vers <base_url>/embeddings,
    l'API OpenAI, au lieu de /api/embed d'Ollama -> 404 systematique ; cf.
    issue GitHub confident-ai/deepeval#1473). On reutilise embeddings.py
    (BGEEmbeddings), deja fonctionnel et coherent avec l'embedding utilise a
    l'ingestion reelle -- pas une deuxieme implementation.

CATEGORIE "unanswerable" -- METHODE (confirmee via la doc DeepEval + inspection
du code installe) : sur les 7 types d'evolution de deepeval (REASONING,
MULTICONTEXT, CONCRETIZING, CONSTRAINED, COMPARATIVE, HYPOTHETICAL, IN_BREADTH),
SEULS 4 "s'accrochent" garantie au contexte fourni (MULTICONTEXT, CONCRETIZING,
CONSTRAINED, COMPARATIVE) -- les 3 autres (REASONING, HYPOTHETICAL, IN_BREADTH)
peuvent deriver au-dela du contexte source. On les utilise ICI expres pour
generer des questions plus complexes/derivantes qu'un mismatch aleatoire pur,
PUIS on les associe explicitement a un AUTRE contexte du corpus (jamais le
contexte d'origine) pour garantir le statut unanswerable -- combine le meilleur
des deux : questions realistes (pas juste "hors-sujet" evident), label fiable
(pas de dependance a ce que le modele reconnaisse lui-meme la non-reponse).

Dependances :
  pip install deepeval chromadb langchain-core langchain-community langchain-text-splitters
"""

import argparse
import json
import random
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.models import OllamaModel, DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.config import FiltrationConfig, StylingConfig, ContextConstructionConfig, EvolutionConfig
from deepeval.synthesizer.types import Evolution

from ..core.embeddings import BGEEmbeddings

GEN_MODEL_FOR_SYNTHESIS = "qwen2.5:32b"   # H100 -- capable de respecter un grounding strict, contrairement a un petit modele
OLLAMA_HOST = "http://127.0.0.1:11434"


class OllamaEmbedderForDeepEval(DeepEvalBaseEmbeddingModel):
    """Contourne le bug confirme de deepeval.models.OllamaEmbeddingModel (cf.
    docstring du module). Reutilise embeddings.BGEEmbeddings TEL QUEL -- meme
    modele, meme methode que l'ingestion reelle, pas de deuxieme implementation."""

    def __init__(self):
        self._emb = BGEEmbeddings()

    def load_model(self):
        return self._emb

    def embed_text(self, text):
        return self._emb.embed_query(text)

    def embed_texts(self, texts):
        return self._emb.embed_documents(texts)

    async def a_embed_text(self, text):
        return self.embed_text(text)

    async def a_embed_texts(self, texts):
        return self.embed_texts(texts)

    def get_model_name(self):
        return "bge-m3 (embeddings.BGEEmbeddings)"


def build_synthesizer(evolution_config=None):
    # temperature=0 (pas 0.3) : ce meme modele genere aussi expected_output, le
    # point le plus sensible pour le grounding strict -- la variete des questions
    # vient surtout des types d'evolution et des chunks differents, pas du hasard
    # token par token, donc pas de raison d'y laisser de la marge de derive.
    model = OllamaModel(model=GEN_MODEL_FOR_SYNTHESIS, base_url=OLLAMA_HOST, temperature=0.1)

    styling_config = StylingConfig(
        task="Repondre a des questions d'etudiants sur un cours de deep learning (d2l.ai, NYU-DLSP21)",
        scenario="Etudiant preparant un examen, posant des questions precises sur le contenu du cours",
        input_format="Question claire et specifique sur un concept du cours, en une phrase",
        expected_output_format=(
            "Reponse factuelle et precise, redigee EXCLUSIVEMENT a partir des informations "
            "explicitement presentes dans le contexte fourni. N'ajoute AUCUNE connaissance "
            "generale, aucun exemple, aucune elaboration absente du contexte -- meme si cette "
            "connaissance est correcte par ailleurs. Si le contexte est incomplet sur un point, "
            "la reponse reste incomplete sur ce point plutot que de combler le vide."
        ),
    )

    filtration_config = FiltrationConfig(
        critic_model=model,
        synthetic_input_quality_threshold=0.6,   # un peu plus strict que le defaut deepeval (0.5)
    )

    return Synthesizer(model=model, styling_config=styling_config, filtration_config=filtration_config,
                        evolution_config=evolution_config)


# evolutions qui NE s'accrochent PAS forcement au contexte fourni (confirme dans
# la doc deepeval) -- utilisees UNIQUEMENT pour generer les candidats a la
# categorie "unanswerable" (cf. generate_drift_candidates), jamais pour les
# questions answerable normales (ou MULTICONTEXT/CONCRETIZING/CONSTRAINED/
# COMPARATIVE -- le defaut de deepeval -- restent plus appropriees, car elles
# garantissent un ancrage au contexte).
DRIFT_EVOLUTION_CONFIG = EvolutionConfig(
    num_evolutions=1,
    evolutions={
        Evolution.REASONING: 0.34,
        Evolution.HYPOTHETICAL: 0.33,
        Evolution.IN_BREADTH: 0.33,
    },
)


def _make_context_config(max_contexts_per_document, min_context_length=1, max_context_length=3):
    """min_context_length/max_context_length controlent le nombre de CHUNKS par
    groupe de contexte -- C'EST ce parametre qui determine single vs multi
    passage (1 chunk = single, 2+ = multi), PAS max_contexts_per_document (qui
    controle le nombre de groupes de contexte extraits, pas leur taille)."""
    return ContextConstructionConfig(
        embedder=OllamaEmbedderForDeepEval(),
        critic_model=OllamaModel(model=GEN_MODEL_FOR_SYNTHESIS, base_url=OLLAMA_HOST, temperature=0),
        max_contexts_per_document=max_contexts_per_document,
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        chunk_size=800,               # proche de CHILD_MAX (chunk_parent_child.py) -- coherence avec le retrieval reel
        context_quality_threshold=0.6,
    )


def _select_documents(corpus_dir, n_target, buffer_factor, seed=42):
    """Si le corpus a plus de documents que necessaire pour le buffer souhaite
    (n_target x buffer_factor), en echantillonne un SOUS-ENSEMBLE plutot que de
    tout traiter -- chunker/embedder/noter en qualite 248 documents pour un
    besoin de 20 questions est du travail inutile, meme avec
    max_contexts_per_document=1 (248 contextes construits pour 20 gardes).
    Renvoie (document_paths a utiliser, max_contexts_per_document)."""
    all_paths = [str(p) for p in Path(corpus_dir).rglob("*.md")]
    if not all_paths:
        raise ValueError(f"aucun fichier .md trouve dans {corpus_dir}")

    desired_total = n_target * buffer_factor
    if len(all_paths) > desired_total:
        rng = random.Random(seed)
        return rng.sample(all_paths, desired_total), 1
    return all_paths, max(1, -(-desired_total // len(all_paths)))  # ceil(desired_total / len(all_paths))


def _generate_exact(corpus_dir, n_target, min_context_length, max_context_length, label, buffer_factor=3):
    """Sur-genere (echantillonne juste assez de documents, cf. _select_documents)
    puis TRONQUE a exactement n_target -- deepeval n'offre pas de parametre
    'genere exactement N'. Avertit si le nombre reellement genere est insuffisant."""
    document_paths, max_contexts_per_document = _select_documents(corpus_dir, n_target, buffer_factor)
    print(f"  {label} : {len(document_paths)} documents utilises (sur le corpus complet), "
          f"max_contexts_per_document={max_contexts_per_document} (cible {n_target}, buffer x{buffer_factor})")

    synthesizer = build_synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        include_expected_output=True,
        max_goldens_per_context=1,   # 1 par groupe de contexte -- max de diversite pour le meme nombre de generations
        context_construction_config=_make_context_config(
            max_contexts_per_document, min_context_length, max_context_length),
    )
    if len(goldens) < n_target:
        print(f"  [avertissement] {label} : seulement {len(goldens)}/{n_target} generes -- "
              f"remonte buffer_factor (corpus insuffisant pour ce volume au filtrage qualite actuel ?)")
    return goldens[:n_target]



def generate_single_passage(corpus_dir, n=20, buffer_factor=3):
    """Contexte force a 1 SEUL chunk (min=max=1) -- garantit single_passage,
    pas une question de chance sur le groupement."""
    return _generate_exact(corpus_dir, n, min_context_length=1, max_context_length=1,
                            label="single_passage", buffer_factor=buffer_factor)


def generate_multi_passage(corpus_dir, n=20, buffer_factor=3):
    """Contexte force a 2-3 chunks -- garantit multi_passage."""
    return _generate_exact(corpus_dir, n, min_context_length=2, max_context_length=3,
                            label="multi_passage", buffer_factor=buffer_factor)


def generate_drift_candidates(corpus_dir, n_candidates=10, buffer_factor=3):
    """Genere des questions via les evolutions REASONING/HYPOTHETICAL/IN_BREADTH
    (pas garanties ancrees au contexte) -- base pour construire la categorie
    unanswerable dans add_unanswerable(). include_expected_output=False : on
    n'utilise jamais le ground_truth genere ici (le contexte final sera de toute
    facon remplace par un autre, cf. add_unanswerable), pas la peine de le calculer."""
    document_paths, max_contexts_per_document = _select_documents(corpus_dir, n_candidates, buffer_factor)
    print(f"  unanswerable : {len(document_paths)} documents utilises, "
          f"max_contexts_per_document={max_contexts_per_document}")

    synthesizer = build_synthesizer(evolution_config=DRIFT_EVOLUTION_CONFIG)
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        include_expected_output=False,
        max_goldens_per_context=1,
        context_construction_config=_make_context_config(max_contexts_per_document),
    )
    if len(goldens) < n_candidates:
        print(f"  [avertissement] unanswerable : seulement {len(goldens)}/{n_candidates} candidats generes -- "
              f"remonte buffer_factor")
    return goldens[:n_candidates]


def add_unanswerable(drift_candidates, answerable_goldens, seed=42):
    """Associe chaque question 'derivante' (generee via generate_drift_candidates,
    evolutions non garanties ancrees) au contexte d'un golden ANSWERABLE pris au
    hasard parmi les autres -- jamais son propre contexte d'origine. Le
    ground_truth est fixe (phrase de refus), pas genere : on ne depend jamais de
    ce que le modele reconnaisse lui-meme la non-reponse, seulement du mismatch
    explicite entre question et contexte fourni."""
    rng = random.Random(seed)
    context_pool = [g.context for g in answerable_goldens if g.context]
    if not context_pool or not drift_candidates:
        return []
    unanswerable = []
    for q in drift_candidates:
        mismatched_context = rng.choice(context_pool)
        unanswerable.append({
            "question": q.input,
            "contexts": mismatched_context,
            "ground_truth": "L'information n'est pas présente dans le contexte.",
            "category": "unanswerable",
        })
    return unanswerable


def goldens_to_records(goldens):
    """Convertit les Golden deepeval au schema CONFIRME utilise par
    evaluate_rag.py (question/contexts/ground_truth/category) -- pas de
    conversion ambigue, ce schema est deja valide en conditions reelles."""
    records = []
    for g in goldens:
        n_ctx = len(g.context) if g.context else 1
        records.append({
            "question": g.input,
            "contexts": g.context or [],
            "ground_truth": g.expected_output,
            "category": "multi_passage" if n_ctx > 1 else "single_passage",
        })
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genere un golden dataset via DeepEval Synthesizer, comptes exacts par categorie.")
    ap.add_argument("corpus_dir", help="dossier du corpus normalise (.md)")
    ap.add_argument("--n-single", type=int, default=20, help="nb exact de questions single_passage")
    ap.add_argument("--n-multi", type=int, default=20, help="nb exact de questions multi_passage")
    ap.add_argument("--n-unanswerable", type=int, default=10, help="nb exact de questions unanswerable")
    ap.add_argument("--buffer-factor", type=int, default=3,
                    help="facteur de sur-provisionnement par categorie avant troncature au compte exact (defaut 3x)")
    ap.add_argument("--out", default="eval/golden_dataset.json")
    args = ap.parse_args()

    single = generate_single_passage(args.corpus_dir, n=args.n_single, buffer_factor=args.buffer_factor)
    print(f"{len(single)}/{args.n_single} questions single_passage")

    multi = generate_multi_passage(args.corpus_dir, n=args.n_multi, buffer_factor=args.buffer_factor)
    print(f"{len(multi)}/{args.n_multi} questions multi_passage")

    answerable = list(single) + list(multi)
    records = goldens_to_records(answerable)

    drift_candidates = generate_drift_candidates(args.corpus_dir, n_candidates=args.n_unanswerable,
                                                  buffer_factor=args.buffer_factor)
    print(f"{len(drift_candidates)}/{args.n_unanswerable} candidats unanswerable")

    unanswerable = add_unanswerable(drift_candidates, answerable)
    records.extend(unanswerable)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(records)} questions au total "
          f"({len(single)} single + {len(multi)} multi + {len(unanswerable)} unanswerable) -> {args.out}")