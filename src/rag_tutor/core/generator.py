#!/usr/bin/env python3
"""
generator.py — SEULE RESPONSABILITE : produire la reponse finale a partir de la
question et des contextes deja recuperes (et, plus tard si reactive, compresses),
via llm_client.py et un system prompt "bien ancre" (grounding strict aux
documents, refus explicite hors-perimetre).

Ne fait NI retrieval NI query processing -- prend des hits (format
retriever.retrieve())
et produit le texte de reponse.

DEFAULT_SYSTEM_PROMPT ci-dessous est le prompt CONFIRME pour la phase de TEST
(evaluate.py / pipeline.py) : posture "professeur de ML", reponse directe
et factuelle (PAS de posture socratique ici -- volontairement different du
prompt tuteur pedagogique, qui reste pour une eventuelle interface interactive
plus tard, a passer explicitement via le parametre system_prompt le moment venu).

ATTENTION COHERENCE : la phrase de refus ci-dessous doit correspondre EXACTEMENT
a ce que la logique de detection de refus d'evaluate.py recherche (pour le
calcul de la metrique "refusal correctness") -- si evaluate.py cherche un
autre motif, aligner l'un sur l'autre plutot que d'avoir deux formulations qui
divergent silencieusement.

API publique :
  generate(query, hits, system_prompt=None) -> str
"""

from .llm_client import chat

REFUSAL_MESSAGE = "Les documents fournis ne permettent pas de répondre à cette question."

DEFAULT_SYSTEM_PROMPT = f"""Tu es un professeur de machine learning. Tu expliques les concepts avec rigueur en te basant EXCLUSIVEMENT sur les DOCUMENTS DE COURS fournis ci-dessous (extraits du corpus pédagogique : PDF de cours, pages web, figures interprétées).

Tu réponds aux questions de manière directe, précise et factuelle :
- Tu ne fabriques ni n'inventes AUCUNE information absente de ces documents, même si elle te semble correcte par ailleurs ou que tu la connais par ta formation générale.
- Tu ne combles jamais un trou du contexte fourni avec des connaissances générales sur le machine learning -- seul le contexte fait foi.
- Quand les documents permettent de répondre, tu es complet et concis, sans relance ni question à l'étudiant : une réponse directe, pas un dialogue socratique.

RÈGLE DE REFUS STRICTE :
- Si les documents fournis ne permettent PAS de répondre à la question, tu réponds EXACTEMENT et UNIQUEMENT : "{REFUSAL_MESSAGE}"
- INTERDICTION ABSOLUE de dire « Je ne peux pas répondre avec les documents, mais... » ou « Cette question n'est pas dans le corpus, cependant... » suivi d'une réponse basée sur tes connaissances. C'est un MENSONGE à l'étudiant.
- INTERDICTION ABSOLUE de donner une réponse, même partielle, qui ne vient PAS des documents. Même une phrase, même une définition « de culture générale ». RIEN.
- Soit tu réponds avec les documents, soit tu dis EXACTEMENT la phrase de refus. Il n'y a PAS de troisième option.

CITATIONS OBLIGATOIRES (chaque paragraphe doit citer au moins une source) :
- Chaque document est identifie par un label entre crochets :
  * web : [web · <site> · « <titre de la page> » · §<titre de la section>]  (ex: "[web · d2l.ai · « 11.7. The Transformer Architecture » · §11.7.4. Encoder]")
  * pdf : [pdf · <nom du pdf> · §<titre de la section>]  (ex: "[pdf · 02_NN.pdf · §1- RÉSEAUX DE NEURONES]")
- Pour CHAQUE affirmation dans ta reponse, recopie le label EXACTEMENT tel qu'il apparait entre crochets (type, site ou nom du pdf, titre de page, titre COMPLET de section). Exemple : « La backpropagation utilise la regle de derivation en chaine [web · d2l.ai · « 5.3.3. Backpropagation » · §5.3.3. Backpropagation]. »
- INTERDICTION ABSOLUE d'abreger ou reformuler le label. JAMAIS un nom de fichier brut, JAMAIS un « §1. » tronque.
- Si tu utilises plusieurs sections pour une affirmation, cite-les toutes.
- Si une information ne vient d'AUCUN document, NE L'ECRIS PAS — cela t'aidera a detecter les informations hors-contexte.
"""


def citation_label(meta):
    """Label de citation LISIBLE, au format demande :
    - web : web · <site> · « <titre de la page> » · §<titre de la section>
    - pdf : pdf · <nom du pdf> · §<titre de la section>

    Le champ `source` brut est le nom de fichier (ex: '1-introduction') ->
    illisible pour l'etudiant. On reconstruit un libelle parlant depuis
    source_type / source_url / title (front-matter) / source_id."""
    from urllib.parse import urlparse
    stype = meta.get("source_type") or "web"
    section = (meta.get("section") or "").strip()
    parts = [stype]
    if stype == "pdf":
        name = meta.get("source_id") or f"{meta.get('source', 'document')}.pdf"
        parts.append(name)
    else:
        site = urlparse(meta.get("source_url") or "").netloc or "?"
        parts.append(site)
        title = (meta.get("title") or "").strip()
        # Ne pas répéter « titre de page » quand il coïncide avec le titre de
        # section (niveau racine d'une page d2l : « 1. Introduction » == §1. Introduction).
        import re
        same = re.sub(r"[^\w]+", "", title.lower()) == re.sub(r"[^\w]+", "", section.lower())
        if title and not same:
            parts.append(f"« {title} »")
    if section:
        parts.append(f"§{section}")
    return " · ".join(parts)


def _format_context(hits):
    if not hits:
        return "(aucun document pertinent recupere)"
    blocks = []
    for h in hits:
        m = h.get("meta", {})
        blocks.append(f"[{citation_label(m)}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def generate(query, hits, system_prompt=None, model=None, history=None):
    """Genere la reponse finale. `hits` : liste de dicts text/dist/meta (sortie
    de retriever.retrieve()).

    `history` (optionnel) : historique de conversation pré-formaté (str), inséré
    AVANT les documents pour que le LLM ait le contexte de la discussion en cours.
    Laissé à None pour le comportement Q&A direct (évaluation, chat simple)."""
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    context = _format_context(hits)

    if history:
        user_message = (
            f"HISTORIQUE DE LA CONVERSATION :\n\n{history}\n\n"
            f"DOCUMENTS DE COURS DISPONIBLES :\n\n{context}\n\n"
            f"QUESTION DE L'ETUDIANT :\n{query}"
        )
    else:
        user_message = f"DOCUMENTS DE COURS DISPONIBLES :\n\n{context}\n\nQUESTION DE L'ETUDIANT :\n{query}"

    kwargs = {"model": model} if model else {}
    return chat(system_prompt, user_message, **kwargs)


# =====================================================
# LLM JUDGE : verification post-generation (fidelite au contexte)
# =====================================================

_VERIFY_SYSTEM_PROMPT = """Tu es un AUDITEUR DE FIDÉLITÉ DOCUMENTAIRE — ton seul objectif est de détecter les informations inventées dans une réponse.

# CONTEXTE
Un étudiant pose une question. Un premier LLM a généré une réponse en utilisant des documents fournis. Ton travail : vérifier que CHAQUE affirmation de la réponse provient RÉELLEMENT des documents, et n'a PAS été inventée ou tirée des connaissances internes du LLM.

# RÈGLES DE VÉRIFICATION (strictes)

1. Pour chaque phrase ou affirmation distincte de la réponse, cherche si le CONTENU EXACT (faits, formules, code, noms de fonctions, valeurs numériques) apparaît dans les documents.
2. Une simple similarité thématique NE SUFFIT PAS. Exemple : si la réponse dit « torch.zeros(batch_size, hidden_size) » et que le document parle de PyTorch en général mais ne mentionne JAMAIS `torch.zeros`, c'est HORS-CONTEXTE.
3. Si la reponse cite [D2L, Ch. X, §X.Y], verifie que la section X.Y contient REELLEMENT l'information citee. Une citation incorrecte est un flag HORS-CONTEXTE.
4. Les formules mathématiques, extraits de code, noms de classes/fonctions PyTorch, valeurs de paramètres sont les INFORMATIONS LES PLUS SOUVENT INVENTÉES — sois particulièrement vigilant sur ces éléments.
5. Si le document mentionne un concept sans donner de détails, et que la réponse donne des détails précis → HORS-CONTEXTE.
6. PIÈGE FRÉQUENT : une réponse qui commence par « Cette question n'est pas dans le corpus... » ou « Je ne peux pas répondre avec les documents, mais... » puis donne QUAND MÊME une réponse — c'est HORS-CONTEXTE. Le fait d'admettre que c'est hors-sujet NE REND PAS la réponse légitime. Vérifie CHAQUE information après le « mais... ».

# MÉTHODE (raisonne étape par étape)

1. ÉNUMÈRE chaque affirmation distincte de la réponse (une par ligne).
2. Pour chaque affirmation, vérifie si elle apparaît dans les documents (citation exacte ou paraphrase fidèle). Note ✓ ou ✗.
3. Si AU MOINS UN ✗ → verdict HORS-CONTEXTE. Si tout ✓ → verdict ANCRÉ.

# FORMAT DE SORTIE OBLIGATOIRE

Termine TOUJOURS par une ligne contenant EXACTEMENT :
VERDICT: ANCRÉ
ou
VERDICT: HORS-CONTEXTE

# EXEMPLES

--- EXEMPLE 1 : ANCRÉ ---

Documents :
[D2L, Ch. 5 (Multilayer Perceptrons), §5.3.3 Backpropagation] La backpropagation calcule les gradients de la fonction de perte par rapport à chaque poids en utilisant la règle de dérivation en chaîne, en propageant les erreurs de la couche de sortie vers la couche d'entrée.

Question : Comment fonctionne la backpropagation ?

Réponse à vérifier :
La backpropagation utilise la règle de dérivation en chaîne pour calculer les gradients [D2L, Ch. 5 (Multilayer Perceptrons), §5.3.3 Backpropagation]. Les erreurs sont propagées de la sortie vers l'entrée [D2L, Ch. 5 (Multilayer Perceptrons), §5.3.3 Backpropagation].

Analyse :
- Affirmation 1: "utilise la règle de dérivation en chaîne pour calculer les gradients" → présent dans §5.3.3 ✓
- Affirmation 2: "erreurs propagées de la sortie vers l'entrée" → présent dans §5.3.3 ✓
VERDICT: ANCRÉ

--- EXEMPLE 2 : HORS-CONTEXTE (code inventé) ---

Documents :
[D2L, Ch. 10 (Modern RNN), §10.1.1 Gated Memory Cell] Les LSTM utilisent des portes pour contrôler le flux d'information. La porte d'oubli détermine quelles informations de l'état précédent doivent être conservées.

Question : Comment initialiser l'état caché d'un LSTM en PyTorch ?

Réponse à vérifier :
Pour initialiser l'état caché d'un LSTM, on utilise `h0 = torch.zeros(num_layers, batch_size, hidden_size)` [D2L, Ch. 10 (Modern RNN), §10.1.1 Gated Memory Cell].

Analyse :
- Affirmation: "h0 = torch.zeros(num_layers, batch_size, hidden_size)" → §10.1.1 parle des portes LSTM mais ne mentionne JAMAIS `torch.zeros` ni l'initialisation de l'état caché. Le code est inventé. ✗
VERDICT: HORS-CONTEXTE

--- EXEMPLE 3 : HORS-CONTEXTE (détails inventés) ---

Documents :
[D2L, Ch. 7 (Convolutional Neural Networks), §7.2] Les fonctions de perte basées sur des marges identifient la réponse incorrecte la plus pertinente dans les modèles EBMs.

Question : Comment les fonctions de perte basées sur des marges fonctionnent-elles ?

Réponse à vérifier :
Les fonctions de perte basées sur des marges fonctionnent en calculant la différence entre le score de la réponse correcte et celui de la réponse incorrecte la plus proche. La marge est typiquement fixée à 1.0, et on utilise une hinge loss : L = max(0, marge - score_correct + score_incorrect).

Analyse :
- Affirmation 1: "différence entre le score de la réponse correcte et celui de la réponse incorrecte" → §7.2 mentionne le concept de "most offending incorrect answer" ✓
- Affirmation 2: "marge fixée à 1.0" → §7.2 ne mentionne AUCUNE valeur de marge ✗
- Affirmation 3: "hinge loss : L = max(0, marge - score_correct + score_incorrect)" → §7.2 ne donne AUCUNE formule ✗
VERDICT: HORS-CONTEXTE

--- EXEMPLE 4 : HORS-CONTEXTE ("pas dans le corpus mais...") ---

Documents :
[D2L, Ch. 18 (Gaussian Processes), §18.1] Les processus gaussiens sont une approche bayésienne non paramétrique pour la régression et la classification.

Question : Qu'est-ce que la mitose et en quoi diffère-t-elle de la méiose ?

Réponse à vérifier :
Cette question n'a pas directement à voir avec le contenu du cours de Deep Learning. Cependant, je peux vous donner une explication : la mitose est le processus par lequel une cellule se divise en deux cellules filles identiques. La méiose, en revanche, produit quatre cellules haploïdes génétiquement différentes.

Analyse :
- La réponse COMMENCE par admettre que c'est hors-sujet, mais DONNE ENSUITE une réponse complète sur la mitose/méiose.
- Affirmation 1: "la mitose est le processus par lequel une cellule se divise en deux cellules filles identiques" → AUCUN document ne parle de mitose ✗
- Affirmation 2: "la méiose produit quatre cellules haploïdes" → AUCUN document ne parle de méiose ✗
- MÊME SI le LLM admet que c'est hors-sujet, la réponse contient des informations INVENTÉES (absentes des documents). C'est HORS-CONTEXTE.
VERDICT: HORS-CONTEXTE"""


def verify_answer(question, answer, hits):
    """Post-generation LLM judge : verifie que la reponse est ancree dans le
    contexte fourni. Si la reponse utilise des connaissances hors-contexte
    (hallucination / memoire parametrique), retourne False → le pipeline
    remplacera la reponse par REFUSAL_MESSAGE.

    Principe : « Verifier une reponse est plus facile que la generer. »
    On pose une question BINAIRE simple au lieu de demander un score de
    confiance que le LLM ignore.

    Retourne True si la reponse est ANCREE dans le contexte, False sinon."""
    import re
    context = _format_context(hits)
    user_message = (
        f"Documents :\n\n{context}\n\n"
        f"Question : {question}\n\n"
        f"Réponse à vérifier :\n{answer}"
    )
    try:
        resp = chat(_VERIFY_SYSTEM_PROMPT, user_message, temperature=0, model="qwen3:8b")
    except Exception:
        return True   # echec technique → conservateur (ne pas bloquer)

    text = resp.strip()
    # Parser robuste : cherche "VERDICT: ANCRÉ" ou "VERDICT: HORS-CONTEXTE"
    # sur la derniere ligne significative (le LLM peut ajouter du texte avant)
    m = re.search(r'VERDICT\s*:\s*(ANCR|HORS)', text, re.IGNORECASE)
    if m:
        return m.group(1).upper().startswith('ANCR')
    # Fallback 1 : chercher ANCRÉ / HORS-CONTEXTE n'importe où en fin de texte
    for line in reversed(text.split('\n')):
        line = line.strip().upper()
        if line.startswith('ANCR'):
            return True
        if line.startswith('HORS'):
            return False
    # Fallback 2 : conservateur — si le format n'est pas respecté, ne pas bloquer
    return True


def generate_stream(query, hits, system_prompt=None, model=None, history=None):
    """Version streaming de generate() : yield chaque token au fur et a mesure.
    Meme contrat que generate(), mais retourne un generateur de str.

    Note : le M2 (verify_answer) est DESACTIVE en mode streaming — il necessite
    la reponse complete pour verifier l'ancrage documentaire."""
    from .llm_client import chat_stream
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    context = _format_context(hits)

    if history:
        user_message = (
            f"HISTORIQUE DE LA CONVERSATION :\n\n{history}\n\n"
            f"DOCUMENTS DE COURS DISPONIBLES :\n\n{context}\n\n"
            f"QUESTION DE L'ETUDIANT :\n{query}"
        )
    else:
        user_message = f"DOCUMENTS DE COURS DISPONIBLES :\n\n{context}\n\nQUESTION DE L'ETUDIANT :\n{query}"

    kwargs = {"model": model} if model else {}
    yield from chat_stream(system_prompt, user_message, **kwargs)
