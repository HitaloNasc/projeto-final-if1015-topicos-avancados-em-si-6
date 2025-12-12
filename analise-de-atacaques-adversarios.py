# %% [markdown]
# # Análise de Ataques Adversários — BERT + Toxicidade
# 
# Entrega 12/12 — Tópicos Avançados em SI 6
# 
# Objetivos deste notebook:
# - Carregar o modelo BERT baseline treinado para toxicidade;
# - Definir ataques adversários simples em textos (typos, sinônimos, etc.);
# - Avaliar como as probabilidades de classes (toxic, insult, threat) mudam;
# - Identificar vulnerabilidades e padrões de comportamento do modelo.


# %% [markdown]
# ## 0) Imports, configuração e labels

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Labels do modelo (Jigsaw multilabel)
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Para a análise de ataques, vamos focar nessas 3:
focus_labels = ["toxic", "insult", "threat"]

device = torch.device("cpu")  # CPU é suficiente aqui
print("Using device:", device)


# %% [markdown]
# ## 1) Carregar modelo baseline e tokenizer

# %%
baseline_path = "baseline_model"

tokenizer = AutoTokenizer.from_pretrained(
    baseline_path,
    use_fast=False,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    baseline_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    local_files_only=True
)

model.to(device)
model.eval()

print("Modelo e tokenizer carregados de:", baseline_path)


# %% [markdown]
# ## 2) Funções auxiliares para predição

# %%
def predict_proba(texts):
    """
    Retorna probabilidades (0-1) para todas as labels, para uma lista de textos.
    """
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()  # [batch, num_labels]
    return probs


def pretty_probs(text, probs, focus_labels=focus_labels):
    """
    Imprime probabilidades de interesse (focus_labels) de forma amigável.
    """
    print(f'Texto: "{text}"')
    out = {}
    for label, prob in zip(label_cols, probs):
        if label in focus_labels:
            out[label] = round(float(prob), 3)
    print("Probabilidades (foco em toxic / insult / threat):", out)


def show_attack(original, attacked, focus_labels=focus_labels):
    """
    Mostra probabilidades antes e depois do ataque adversário.
    """
    probs = predict_proba([original, attacked])
    p_orig = probs[0]
    p_att = probs[1]

    print("\n==================================================")
    print("TEXTO ORIGINAL")
    pretty_probs(original, p_orig, focus_labels)

    print("\nTEXTO ATACADO")
    pretty_probs(attacked, p_att, focus_labels)

    print("\nDiferença (atacado - original):")
    diffs = {}
    for i, label in enumerate(label_cols):
        if label in focus_labels:
            diffs[label] = round(float(p_att[i] - p_orig[i]), 3)
    print(diffs)
    print("==================================================")


# %% [markdown]
# ## 3) Textos base para os ataques
# 
# Vamos testar ataques em dois cenários principais:
# - Um texto de insulto;
# - Um texto de ameaça.

# %%
text_insult = "You are a disgusting idiot and nobody likes you."
text_threat = "I will find you and hurt you badly."

print("Predições base (sem ataque):")
probs_base = predict_proba([text_insult, text_threat])
print("\nTexto de insulto:")
pretty_probs(text_insult, probs_base[0])

print("\nTexto de ameaça:")
pretty_probs(text_threat, probs_base[1])


# %% [markdown]
# ## 4) Definição dos ataques adversários
# 
# Vamos implementar alguns ataques simples:
# 
# 1. **Typos / Leetspeak** – trocar letras por números/símbolos;
# 2. **Sinônimos** – substituir por termos equivalentes;
# 3. **Inserção de palavras neutras** – "poluir" o texto com palavras irrelevantes;
# 4. **Espaçamento/pontuação** – quebrar palavras, adicionar pontos;
# 5. **Reescrita leve** – mudar um pouco a estrutura mantendo o sentido.


# %% [markdown]
# ### 4.1 Ataque de typos / leetspeak

# %%
def typo_attack(text: str) -> str:
    """
    Substitui letras por números/símbolos para tentar enganar o modelo
    sem alterar muito o significado para humanos.
    """
    replacements = {
        "i": "1",
        "o": "0",
        "e": "3",
        "a": "@",
        "u": "ü",
    }
    out = []
    for ch in text:
        lower = ch.lower()
        if lower in replacements:
            # mantém maiúscula/minúscula de forma simples
            rep = replacements[lower]
            out.append(rep)
        else:
            out.append(ch)
    return "".join(out)


# %% [markdown]
# ### 4.2 Ataque por sinônimos (simplificado com dicionário estático)

# %%
synonym_map = {
    "idiot": "fool",
    "disgusting": "gross",
    "hurt": "harm",
    "badly": "severely",
    "nobody": "no one",
}

def synonym_attack(text: str) -> str:
    words = text.split()
    new_words = []
    for w in words:
        key = w.lower().strip(".,!?")
        if key in synonym_map:
            # mantém pontuação final simples
            punct = ""
            if w.endswith((".", ",", "!", "?")):
                punct = w[-1]
            new_words.append(synonym_map[key] + punct)
        else:
            new_words.append(w)
    return " ".join(new_words)


# %% [markdown]
# ### 4.3 Ataque por inserção de palavras neutras

# %%
def insertion_attack_insult(text: str) -> str:
    """
    Insere palavras neutras / hesitações no meio do insulto.
    Ex.: "You are a disgusting idiot" -> "You are, like, a kind of disgusting idiot actually"
    """
    return text.replace(
        "a disgusting idiot",
        "kinda, to be honest, a disgusting idiot actually"
    )


def insertion_attack_threat(text: str) -> str:
    """
    Amacia a forma da ameaça, inserindo "I think", "maybe", etc.
    """
    return text.replace(
        "I will find you and hurt you badly",
        "Maybe I will, you know, find you someday and kind of hurt you, maybe not so badly"
    )


# %% [markdown]
# ### 4.4 Ataque de espaçamento/pontuação

# %%
def spacing_attack(text: str) -> str:
    """
    Quebra palavras ofensivas com espaços ou pontos.
    Ex.: 'idiot' -> 'i d i o t'
    """
    out = text
    out = out.replace("idiot", "i d i o t")
    out = out.replace("hurt you badly", "hurt. you. badly")
    return out


# %% [markdown]
# ### 4.5 Ataque de reescrita leve (paráfrase simples)

# %%
def rewrite_attack_insult(text: str) -> str:
    """
    Reescreve levemente o insulto mantendo o sentido.
    """
    return "People really think you are gross and kind of a fool."

def rewrite_attack_threat(text: str) -> str:
    """
    Reescreve levemente a ameaça mantendo a intenção.
    """
    return "One day I might seriously harm you."


# %% [markdown]
# ## 5) Executando ataques no texto de insulto

# %%
print("\n\n########## ATAQUES NO TEXTO DE INSULTO ##########")

# 1) Typos / leetspeak
att_insult_typos = typo_attack(text_insult)
show_attack(text_insult, att_insult_typos)

# 2) Sinônimos
att_insult_syn = synonym_attack(text_insult)
show_attack(text_insult, att_insult_syn)

# 3) Inserção de palavras neutras / hesitação
att_insult_insert = insertion_attack_insult(text_insult)
show_attack(text_insult, att_insult_insert)

# 4) Espaçamento / pontuação
att_insult_space = spacing_attack(text_insult)
show_attack(text_insult, att_insult_space)

# 5) Reescrita leve
att_insult_rewrite = rewrite_attack_insult(text_insult)
show_attack(text_insult, att_insult_rewrite)


# %% [markdown]
# ## 6) Executando ataques no texto de ameaça

# %%
print("\n\n########## ATAQUES NO TEXTO DE AMEAÇA ##########")

# 1) Typos / leetspeak
att_threat_typos = typo_attack(text_threat)
show_attack(text_threat, att_threat_typos)

# 2) Sinônimos
att_threat_syn = synonym_attack(text_threat)
show_attack(text_threat, att_threat_syn)

# 3) Inserção de palavras que "amaciam" a frase
att_threat_insert = insertion_attack_threat(text_threat)
show_attack(text_threat, att_threat_insert)

# 4) Espaçamento / pontuação
att_threat_space = spacing_attack(text_threat)
show_attack(text_threat, att_threat_space)

# 5) Reescrita leve
att_threat_rewrite = rewrite_attack_threat(text_threat)
show_attack(text_threat, att_threat_rewrite)


# %% [markdown]
# ## 7) Comentários e apontamentos para o relatório
# 
# A partir dos resultados impressos acima, você deve observar:
# 
# - Em quais ataques a probabilidade de **toxic** cai muito (ex.: typos ou leetspeak);
# - Se o modelo continua detectando **insult** quando troca "idiot" por "fool";
# - Se a detecção de **threat** permanece estável em paráfrases da ameaça;
# - Se ataques de espaçamento ("i d i o t") reduzem muito a toxicidade;
# - Se inserções de palavras neutras mudam pouco ou muito a predição.
# 


# %% [markdown]
# ## 7) Consolidar resultados dos ataques em tabela (DataFrame)

# %%
import pandas as pd

scenarios_insult = {
    "original": text_insult,
    "typos_leet": typo_attack(text_insult),
    "synonyms": synonym_attack(text_insult),
    "insertion": insertion_attack_insult(text_insult),
    "spacing": spacing_attack(text_insult),
    "rewrite": rewrite_attack_insult(text_insult),
}

scenarios_threat = {
    "original": text_threat,
    "typos_leet": typo_attack(text_threat),
    "synonyms": synonym_attack(text_threat),
    "insertion": insertion_attack_threat(text_threat),
    "spacing": spacing_attack(text_threat),
    "rewrite": rewrite_attack_threat(text_threat),
}

rows_insult = []
for name, txt in scenarios_insult.items():
    probs = predict_proba([txt])[0]
    row = {
        "cenario": name,
        "texto": txt,
        "toxic": float(probs[label_cols.index("toxic")]),
        "insult": float(probs[label_cols.index("insult")]),
        "threat": float(probs[label_cols.index("threat")]),
        "tipo_base": "insult",
    }
    rows_insult.append(row)

rows_threat = []
for name, txt in scenarios_threat.items():
    probs = predict_proba([txt])[0]
    row = {
        "cenario": name,
        "texto": txt,
        "toxic": float(probs[label_cols.index("toxic")]),
        "insult": float(probs[label_cols.index("insult")]),
        "threat": float(probs[label_cols.index("threat")]),
        "tipo_base": "threat",
    }
    rows_threat.append(row)

df_attacks = pd.DataFrame(rows_insult + rows_threat)
df_attacks


# %% [markdown]
# ## 8) Gráficos — Ataques no texto de insulto

# %%
import matplotlib.pyplot as plt

df_insult = df_attacks[df_attacks["tipo_base"] == "insult"].copy()

cenarios = df_insult["cenario"].tolist()
x = range(len(cenarios))

plt.figure()
plt.xticks(x, cenarios, rotation=45)
plt.ylabel("Probabilidade")
plt.title("Probabilidades para o texto de insulto (original vs ataques)")

plt.plot(x, df_insult["toxic"].tolist(), marker="o", label="toxic")
plt.plot(x, df_insult["insult"].tolist(), marker="o", label="insult")
plt.plot(x, df_insult["threat"].tolist(), marker="o", label="threat")

plt.tight_layout()
plt.legend()
plt.show()


# %% [markdown]
# ## 9) Gráficos — Ataques no texto de ameaça

# %%
df_threat = df_attacks[df_attacks["tipo_base"] == "threat"].copy()

cenarios = df_threat["cenario"].tolist()
x = range(len(cenarios))

plt.figure()
plt.xticks(x, cenarios, rotation=45)
plt.ylabel("Probabilidade")
plt.title("Probabilidades para o texto de ameaça (original vs ataques)")

plt.plot(x, df_threat["toxic"].tolist(), marker="o", label="toxic")
plt.plot(x, df_threat["insult"].tolist(), marker="o", label="insult")
plt.plot(x, df_threat["threat"].tolist(), marker="o", label="threat")

plt.tight_layout()
plt.legend()
plt.show()


