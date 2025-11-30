# %% [markdown]
# # Análise de Robustez — Jigsaw Toxicity + BERT
# 
# Neste notebook:
# - carregamos o modelo baseline salvo,
# - preparamos uma amostra de validação,
# - aplicamos perturbações nos textos,
# - medimos a queda de F1 para cada tipo de ruído.


# %% [markdown]
# ## 0) Imports e configuração de device

# %%
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt
import pandas as pd

# Detectar device (MPS ou CPU)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# %% [markdown]
# ## 1) Carregar modelo baseline e tokenizer

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

baseline_path = "baseline_model"

# Força tokenizer "lento" para evitar crash do tokenizers (Rust) no Python 3.14
tokenizer = AutoTokenizer.from_pretrained(
    baseline_path,
    use_fast=False,
    local_files_only=True
)

# Para robustez (500 exemplos), CPU é suficiente e mais estável
device = torch.device("cpu")
print("Using device:", device)

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
# ## 2) Carregar dataset e preparar amostra para robustez

# %%
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

dataset = load_dataset(
    "csv",
    data_files={"train": "dataset/train.csv"}
)

def add_labels(example):
    example["labels"] = [example[c] for c in label_cols]
    return example

dataset["train"] = dataset["train"].map(add_labels)

# Split para obter uma "validação"
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
val_ds = split["test"]

# Para robustez, usamos só uma amostra (para ficar leve e rápido)
sample_size = 500
val_sample = val_ds.shuffle(seed=42).select(range(sample_size))

texts = val_sample["comment_text"]
labels = np.array(val_sample["labels"])

print("Amostra para robustez:", len(texts), "exemplos")


# %% [markdown]
# ## 3) Função auxiliar de avaliação (F1 em lista de textos)

# %%
def evaluate_texts(text_list, batch_size=32):
    """Avalia uma lista de textos e retorna as predições (0/1) e F1 micro."""
    all_preds = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.append(preds)

    all_preds = np.vstack(all_preds)
    f1 = f1_score(labels, all_preds, average="micro")
    return all_preds, f1


# %% [markdown]
# ## 4) Perturbações de texto para testar robustez

# %%
random.seed(42)

def add_typos(text, prob=0.15):
    """Introduce random character substitutions em letras."""
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < prob:
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def add_emojis(text):
    """Adiciona um emoji de raiva ao final do texto."""
    return text + " 😡"


def mask_offensive_chars(text):
    """Substitui algumas palavras ofensivas por eufemismos/leves censuras."""
    replacements = {
        "idiot": "id!ot",
        "stupid": "stu_pid",
        "bitch": "b!tch",
        "fuck": "f*ck",
        "asshole": "a**hole",
    }
    lowered = text.lower()
    for k, v in replacements.items():
        lowered = lowered.replace(k, v)
    # Mantém caixa original simples (poderia ser melhorado, mas serve para a análise)
    return lowered


def light_shuffle(text):
    """Faz uma pequena troca de posição entre duas palavras internas."""
    words = text.split()
    if len(words) > 3:
        i = random.randint(1, len(words) - 2)
        j = random.randint(1, len(words) - 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


# %% [markdown]
# ## 5) F1 baseline (sem perturbação)

# %%
baseline_preds, f1_base = evaluate_texts(texts)
print("F1 baseline:", f1_base)


# %% [markdown]
# ## 6) F1 com diferentes perturbações

# %%
# Typos
texts_typos = [add_typos(t) for t in texts]
_, f1_typos = evaluate_texts(texts_typos)
print("F1 com typos:", f1_typos)

# Emojis
texts_emojis = [add_emojis(t) for t in texts]
_, f1_emojis = evaluate_texts(texts_emojis)
print("F1 com emojis:", f1_emojis)

# Eufemismos / mascarar palavras
texts_mask = [mask_offensive_chars(t) for t in texts]
_, f1_mask = evaluate_texts(texts_mask)
print("F1 com eufemismos/mascara:", f1_mask)

# Shuffle leve
texts_shuffle = [light_shuffle(t) for t in texts]
_, f1_shuffle = evaluate_texts(texts_shuffle)
print("F1 com shuffle leve:", f1_shuffle)


# %% [markdown]
# ## 7) Tabela resumida de robustez

# %%
results = pd.DataFrame([
    ["Baseline",  f1_base,   0.0],
    ["Typos",     f1_typos,  f1_typos  - f1_base],
    ["Emojis",    f1_emojis, f1_emojis - f1_base],
    ["Eufemismos",f1_mask,   f1_mask   - f1_base],
    ["Shuffle",   f1_shuffle,f1_shuffle- f1_base],
], columns=["Perturbação", "F1", "Delta F1"])

print(results)


# %% [markdown]
# ## 8) Gráfico de F1 por tipo de perturbação

# %%
plt.figure(figsize=(8, 4))
plt.bar(results["Perturbação"], results["F1"])
plt.title("Robustez do Modelo — F1 por Perturbação")
plt.ylabel("F1 Micro")
plt.xlabel("Perturbação")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.show()


# %% [markdown]
# ## 9) Observação (para o relatório)
# 
# A partir da tabela e do gráfico, você pode descrever no relatório:
# - quais perturbações causam maior queda de F1,
# - em quais casos o modelo é mais robusto,
# - hipóteses do porquê (ex.: emojis quase não mudam o significado,
#   já eufemismos ou typos podem "enganar" o vocabulário do BERT).
