# 📄 Relatório da Entrega — 05/12
## Análise de Interpretabilidade do Modelo BERT para Classificação de Toxicidade
### Tópicos Avançados em Sistemas de Informação (IF1015)

---

Este relatório apresenta a avaliação de interpretabilidade de um modelo BERT treinado para classificação multilabel de toxicidade (dataset Jigsaw Toxicity). São aplicadas duas técnicas de explicabilidade:

- **LIME (Local Interpretable Model-Agnostic Explanations)** — explicações locais baseadas em perturbação do texto.  
- **Integrated Gradients (Captum)** — explicações baseadas em gradiente sobre as embeddings do modelo.

As técnicas permitem entender *quais tokens mais contribuem* para cada predição, verificando se o modelo aprendeu padrões linguísticos coerentes.

---

# 1. Amostra de texto analisado

O texto selecionado para análise com LIME foi:

> **"You are a disgusting idiot and nobody likes you."**

O texto analisado com Integrated Gradients foi:

> **"I will find you and hurt you badly."**

---

# 2. Explicabilidade com LIME

LIME identifica os tokens que mais influenciam a probabilidade prevista para uma determinada classe. As tabelas abaixo mostram os pesos atribuídos a cada token — valores positivos indicam que o token **aumenta** a probabilidade da classe, e negativos indicam que **reduz**.

---

## 2.1 Explicação da classe **"insult"**

- idiot -> 0.4289
- disgusting -> 0.1621
- You -> 0.0603
- you -> 0.0368
- and -> -0.0293
- a -> 0.0276
- nobody -> 0.0153
- likes -> -0.0112
- are -> 0.0024


**Interpretação**

- Os tokens mais relevantes para o modelo detectar *insulto* foram **idiot** e **disgusting**, o que demonstra aprendizado adequado de termos ofensivos diretos.
- Pronome **you / You** também aumenta a probabilidade, pois personaliza a agressão.
- Conectivos como **and**, **likes** apresentam impacto pequeno ou negativo — esperado, pois não carregam conteúdo ofensivo.

Arquivo gerado: **`lime_insult.html`**

---

## 2.2 Explicação da classe **"toxic"**

- idiot -> 0.3505
- disgusting -> 0.2075
- You -> 0.0483
- nobody -> 0.0384
- you -> 0.0355
- a -> 0.0266
- and -> -0.0222
- are -> 0.0154
- likes -> -0.0009


**Interpretação**

- Novamente, **idiot** e **disgusting** dominam como indicadores de toxicidade geral.
- Tokens **You**, **you**, **nobody** contribuem positivamente por estarem associados a ataques pessoais ou linguagem depreciativa.
- Palavras funcionais têm impacto desprezível ou negativo, o que indica boa separação semântica aprendida pelo modelo.

---

# 3. Explicabilidade com Integrated Gradients (Captum)

Integrated Gradients mede a contribuição de cada token calculando o gradiente entre um baseline neutro e a entrada real. Aqui, as explicações foram feitas diretamente sobre as **embeddings** do BERT, garantindo derivabilidade.

### Texto analisado:
> **"I will find you and hurt you badly."**

---

## 3.1 Importância dos tokens (IG)

- [CLS] -> -0.0686
- i -> 0.1119
- will -> 0.2118
- find -> 0.1446
- you -> 0.5793
- and -> -0.1841
- hurt -> 0.3828
- you -> 0.7625
- badly -> 0.0416
- . -> -0.0319
- [SEP] -> 0.2951


**Interpretação**

- O modelo identifica corretamente que **"you"**, **"hurt"**, e **"you" (segunda ocorrência)** são os tokens *mais importantes* para detectar ameaça (*threat*).
- Verbos de intenção (**will**, **find**) também têm forte contribuição.
- Tokens estruturais (**[CLS]**, **[SEP]**, **.**) apresentam influência baixa ou moderada — comportamento normal em modelos BERT.
- O token **"and"** possui peso negativo, sugerindo que o conector suaviza a agressão quando considerado isoladamente.

---

# 4. Conclusões

A análise de interpretabilidade mostra que:

- O modelo **aprendeu padrões semânticos coerentes** com toxicidade e insulto.  
- Palavras ofensivas receberam altos pesos em LIME (idiot, disgusting).  
- Tokens relacionados a ameaça receberam altos pesos no IG (hurt, you, will).  
- Palavras funcionais exibiram pouca influência, indicando que o modelo não está enviesado por estrutura gramatical.  
- O comportamento do modelo é consistente e interpretável, apoiando sua confiabilidade para uso acadêmico e experimental.
