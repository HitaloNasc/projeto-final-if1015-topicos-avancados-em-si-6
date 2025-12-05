# %% [markdown]
# # Análise de Interpretabilidade — BERT + Toxicity
# 
# Entrega 05/12 — Tópicos Avançados em SI 6
# 
# Este notebook realiza:
# - carregamento do modelo baseline salvo (`baseline_model/`);
# - explicações com LIME (Local Interpretable Model-Agnostic Explanations);
# - explicações com Integrated Gradients (Captum);
# - visualizações coloridas (heatmaps) dos tokens mais relevantes.
# 
# O objetivo é entender como o modelo decide suas predições de toxicidade.


# %% [markdown]
# ## 0) Imports e configuração do ambiente

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# CPU = mais estável para interpretabilidade
device = torch.device("cpu")
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
    Retorna probabilidades para cada label (0-1).
    """
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def print_predictions(texts):
    """
    Imprime predições de forma amigável.
    """
    probs = predict_proba(texts)
    for text, p in zip(texts, probs):
        print("\nTexto:", text)
        out = {label: round(float(prob), 3) for label, prob in zip(label_cols, p)}
        print("Predições:", out)


# %% [markdown]
# ## 3) Amostra de textos para interpretação

# %%
samples = [
    "I love this article, very helpful and kind.",
    "You are a disgusting idiot and nobody likes you.",
    "I will find you and hurt you badly.",
    "Thank you for your support!"
]

print_predictions(samples)


# %% [markdown]
# ## 4) Interpretabilidade com LIME (Local Explanations)

# %%
# Instalação (caso não tenha):
# !pip install lime

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=label_cols)


# %% [markdown]
# ### 4.1 Função de predição no formato LIME

# %%
def lime_predict(texts):
    return predict_proba(texts)


# %% [markdown]
# ### 4.2 Gerar explicação LIME para texto ofensivo

# %%
sample_text = samples[1]
print("Texto selecionado:\n", sample_text)

exp = explainer.explain_instance(
    sample_text,
    lime_predict,
    num_features=10,
    top_labels=6
)


# %% [markdown]
# ### 4.2.1 Explicação da classe "insult"

# %%
insult_idx = label_cols.index("insult")

insult_exp = exp.as_list(label=insult_idx)
print("\nExplicação LIME para 'insult':")
for word, weight in insult_exp:
    print(f"{word:15s} -> {weight:.4f}")

exp.save_to_file("lime_insult.html")
print("\nArquivo salvo: lime_insult.html")


# %% [markdown]
# ### 4.3 Explicação da classe "toxic"

# %%
toxic_idx = label_cols.index("toxic")

toxic_exp = exp.as_list(label=toxic_idx)
print("\nExplicação LIME para 'toxic':")
for word, weight in toxic_exp:
    print(f"{word:15s} -> {weight:.4f}")


# %% [markdown]
# ## 5) Interpretabilidade com Integrated Gradients (Captum)

# %%
# Instalação caso necessário:
# !pip install captum

from captum.attr import IntegratedGradients
import torch.nn.functional as F
from IPython.display import HTML


# %% [markdown]
# ### 5.1 Função de forward em cima das embeddings
# 
# Aqui usamos as *embeddings* como entrada contínua para o IG,
# pois não faz sentido derivar em relação a índices inteiros (input_ids).

# %%
def forward_ig(inputs_embeds, attention_mask, target_idx):
    """
    Forward usado pelo Integrated Gradients.
    Recebe embeddings já prontos (float) e retorna
    apenas o logit da classe alvo.
    """
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )
    # logits: [batch, num_labels]
    return outputs.logits[:, target_idx]

ig = IntegratedGradients(forward_ig)



# %% [markdown]
# ### 5.2 Integrated Gradients para texto de ameaça

# %%
threat_text = samples[2]
print("Texto de ameaça:\n", threat_text)

enc = tokenizer(
    threat_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=64
)

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

# gera embeddings de entrada a partir dos input_ids
embedding_layer = model.get_input_embeddings()
inputs_embeds = embedding_layer(input_ids)  # shape: [1, seq_len, hidden_dim]

target_idx = label_cols.index("threat")

# baseline = embeddings nulas (mesmo shape)
baseline_embeds = torch.zeros_like(inputs_embeds)

attributions, delta = ig.attribute(
    inputs=inputs_embeds,
    baselines=baseline_embeds,
    additional_forward_args=(attention_mask, target_idx),
    return_convergence_delta=True
)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# soma as contribuições em cada dimensão do embedding -> 1 valor por token
token_importances = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

print("\nImportância dos tokens:")
for tok, score in zip(tokens, token_importances):
    print(f"{tok:15s} -> {score:.4f}")


## %% [markdown]
# ### 5.3 Visualização colorida dos tokens (HTML)

# %%
def color_token(token, score):
    # normalização simples só para ajustar a transparência
    norm = score / (abs(score) + 1e-9)

    # verde = positivo, vermelho = negativo
    if score >= 0:
        r, g = 0, 180
    else:
        r, g = 180, 0
    opacity = min(0.4, abs(score) * 2)

    return f"<span style='background-color: rgba({r},{g},0,{opacity}); padding:2px; margin:1px;'>{token}</span>"

html_tokens = " ".join(color_token(t, s) for t, s in zip(tokens, token_importances))

HTML(f"<p>{html_tokens}</p>")



# %% [markdown]
# ## 6) Conclusão da interpretabilidade
# 
# - O LIME destacou tokens ofensivos como **idiot**, **disgusting** e **you** como fortes indicadores de toxicidade.
# - O Captum (IG) mostrou que palavras como **hurt**, **badly**, **you** são essenciais para detectar ameaças.
# - Ambas as abordagens revelam que o modelo aprendeu padrões semânticos coerentes.
# - Conectivos gramaticais (*and*, *are*, *likes*) têm baixa ou nenhuma influência.
# - A interpretabilidade confirma que o modelo foca nas partes corretas do texto — essencial para auditoria e confiabilidade.
