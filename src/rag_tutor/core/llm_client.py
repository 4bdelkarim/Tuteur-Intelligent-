#!/usr/bin/env python3
"""
llm_client.py — SEULE RESPONSABILITE : appeler le LLM de generation (Qwen2.5 via
Ollama) avec un system prompt + un message utilisateur, et renvoyer le texte.

Utilise par query_processing.py (reformulation/decomposition) ET generator.py
(reponse finale) -- UN SEUL point d'entree, pour ne jamais avoir deux facons
differentes d'appeler Ollama qui pourraient diverger (modele, temperature...).

IMPORTANT : evaluate_rag.py (le JUGE) ne doit JAMAIS appeler ce module avec
GEN_MODEL -- le bug JUDGE_MODEL/GEN_MODEL confondus sur Colab vient de la, donc
le juge doit passer son propre modele explicitement via le parametre `model`
de chat(), jamais en dependant du defaut ci-dessous.

Client EXPLICITE (ollama.Client(host=...)), pas la fonction globale ollama.chat() :
celle-ci resout son host via OLLAMA_HOST (variable d'env) ou un defaut interne a
la lib, qui peut pointer ailleurs sans prevenir (port herite d'un tunnel ngrok
Colab, instance orpheline sur un port aleatoire...). Avec un Client explicite,
AUCUNE ambiguite sur le serveur contacte -- cf. embeddings.py, meme principe.
"""

GEN_MODEL = "qwen2.5:14b"   # aligne sur les modeles pulles sur Colab (differe du JUDGE_MODEL d'evaluate_rag.py)
MAX_TOKENS = 900             # H100 disponible desormais -- 400 (contrainte historique CPU-only/Colab) coupait
                              # les reponses en plein mot (confirme via chat_cli.py, ex. "...produit la pr").
                              # 900 laisse la place a une explication complete sans etre demesure ; remonter
                              # encore si des reponses continuent d'etre tronquees sur des questions complexes.
NUM_CTX = 24576              # contexte reduit de 32768 (defaut modele) a 24576 : economise ~25% de KV cache
                              # par slot, stocke ~80K caracteres (~25K tokens en moyenne), suffisant pour les
                              # parents retrieves (max ~80K caracteres/parent). Si des reponses sont tronquees
                              # en amont (contexte coupe avant la generation), remonter a 32768.
OLLAMA_HOST = "http://127.0.0.1:11434"   # instance Ollama principale -- port 11434 confirme actif.
                              # cf. embeddings.py, meme raison


def chat(system_prompt, user_message, model=GEN_MODEL, temperature=0.2, max_tokens=MAX_TOKENS,
         host=OLLAMA_HOST, num_ctx=NUM_CTX, keep_alive="30m"):
    """Appel simple : system + user -> texte de reponse (pas de streaming, pas d'historique).

    Parametres
    ----------
    num_ctx : int
        Taille de la fenetre de contexte en tokens (passe dans les options Ollama).
        Reduit de 32768 (defaut modele) a 24576 par defaut pour economiser du KV
        cache. Mettre None pour utiliser le defaut du modele."""
    import ollama
    client = ollama.Client(host=host)
    opts = {"temperature": temperature, "num_predict": max_tokens}
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        options=opts,
    )
    return resp["message"]["content"]

def chat_stream(system_prompt, user_message, model=GEN_MODEL, temperature=0.2,
                max_tokens=MAX_TOKENS, host=OLLAMA_HOST, num_ctx=NUM_CTX,
                keep_alive="30m"):
    """Version streaming de chat() : yield chaque token des qu'il est genere par Ollama.
    Meme signature que chat(), mais retourne un generateur de str.

    Usage :
        for token in chat_stream(system, user):
            print(token, end="", flush=True)
    """
    import ollama
    client = ollama.Client(host=host)
    opts = {"temperature": temperature, "num_predict": max_tokens}
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        options=opts,
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
