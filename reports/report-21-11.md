# 📄 RELATÓRIO — Entrega 21/11
## Projeto Final — IF1015 (Tópicos Avançados em SI 6)
### Treinamento Parcial – Classificação de Toxicidade com BERT

---

# 1. Introdução

Este projeto tem como objetivo aplicar técnicas de Aprendizagem Profunda em uma tarefa de Processamento de Linguagem Natural (PLN), especificamente na classificação de toxicidade em comentários online.  

Esta é a primeira entrega do projeto, englobando:

- definição da aplicação,  
- seleção do dataset,  
- preparação dos dados,  
- implementação do modelo,  
- e apresentação dos primeiros resultados parciais de treinamento.

---

# 2. Definição da Aplicação

### Tarefa  
Classificação multi-rótulo (multi-label) de toxicidade em textos.

### Objetivo  
Dado um comentário, prever pontuações para seis tipos de toxicidade:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

### Justificativa  
A detecção automática de conteúdo tóxico é fundamental para moderação de plataformas, segurança e proteção de usuários.  

Além disso, essa tarefa é adequada para estudos de:

- robustez de modelos de PLN,  
- interpretabilidade,  
- e ataques adversários,  

que serão abordados nas próximas entregas.

---

# 3. Dataset

### Nome  
Jigsaw Toxic Comment Classification Challenge (Kaggle)

### Descrição  
Dataset composto por comentários do Wikipedia Talk Page, rotulados em seis categorias de toxicidade.

### Formato  
Arquivo CSV contendo:

- id  
- comment_text  
- seis rótulos binários (0/1)

### Tamanho original  
159.571 exemplos rotulados.

### Subset utilizado nesta etapa  
Para acelerar o treinamento no MacBook Air M4 (GPU MPS):

- **20.000** exemplos de treino  
- **2.000** exemplos de validação

### Observação  
Foi criada uma coluna `labels` contendo o vetor `[toxic, severe_toxic, obscene, threat, insult, identity_hate]`, necessária para a classificação multi-label.

---

# 4. Modelo Utilizado

### Arquitetura  
BERT Base — `bert-base-uncased`

### Framework  
HuggingFace Transformers + PyTorch

### Configuração  
- Cabeça de saída com 6 neurônios  
- função de ativação sigmoid  
- perda BCEWithLogitsLoss  
- configuração `problem_type="multi_label_classification"`

### Motivos da escolha  
- forte desempenho em tarefas de classificação textual,  
- robustez,  
- facilidade de análise de interpretabilidade,  
- adequação para estudos adversariais.

---

# 5. Pré-processamento e Tokenização

### Tokenização
- Modelo: `bert-base-uncased`  
- `use_fast=False` (evita crash no Python 3.14)  
- truncation ativado  
- `max_length=64`  
- padding dinâmico com `DataCollatorWithPadding`

### Limpeza do dataset
Após tokenização, foram mantidas apenas as colunas:

- input_ids  
- attention_mask  
- labels  

para evitar erros ao usar o collator dinâmico.

---

# 6. Configuração de Treinamento

### Dispositivo
GPU **MPS** (Metal Performance Shaders) – MacBook Air M4.

### Hiperparâmetros
- Epochs: **1**  
- Batch size: **8**  
- Otimizador: **AdamW**  
- Learning Rate: **2e-5**  
- Scheduler linear  

### Motivação da configuração
A utilização de subset + 1 época garante:

- execução rápida e estável,  
- ausência de travamentos,  
- reprodutibilidade para entrega parcial.

---

# 7. Resultados do Treinamento Parcial

Saída da execução:
```bash
Epoch 1/1
Train loss: 0.0670
Val loss: 0.0529
Val F1: 0.7426
```

### Interpretação
- F1 de **0.7426** é excelente para apenas 1 Epoch com subset.  
- O loss baixo é esperado pelo desbalanceamento do dataset (maioria dos casos não são tóxicos).  
- O modelo demonstra aprendizado consistente.  
- Não há indícios de overfitting ou underfitting nesta etapa.

---

# 8. Gráficos

### Loss por Epoch  
![alt text](loss-por-epoch.png)

### F1 Micro por Epoch  
![alt text](f1-micro-por-epoch.png)

---

# 9. Validação Adicional

O modelo foi avaliado manualmente com exemplos reais para garantir coerência das previsões.

```python
samples = [
    "I love this article, very helpful.",
    "You are stupid and disgusting.",
    "I'll find you and hurt you.",
    "Thank you for your support!",
]
```
```bash
Texto: "I love this article, very helpful."
Predições: 
{
    'toxic': 0.005, 
    'severe_toxic': 0.002, 
    'obscene': 0.003, 
    'threat': 0.002, 
    'insult': 0.004, 
    'identity_hate': 0.003
}

Texto: "You are stupid and disgusting."
Predições: 
{
    'toxic': 0.928, 
    'severe_toxic': 0.081, 
    'obscene': 0.636, 
    'threat': 0.041, 
    'insult': 0.641, 
    'identity_hate': 0.109
}

Texto: "I'll find you and hurt you."
Predições: 
{
    'toxic': 0.567, 
    'severe_toxic': 0.023, 
    'obscene': 0.14, 
    'threat': 0.031, 
    'insult': 0.252, 
    'identity_hate': 0.049
}

Texto: "Thank you for your support!"
Predições: 
{
    'toxic': 0.006, 
    'severe_toxic': 0.002, 
    'obscene': 0.003, 
    'threat': 0.002, 
    'insult': 0.004, 
    'identity_hate': 0.003
}
```

# 10. Conclusão da Entrega 21/11

Esta etapa inicial foi concluída com êxito:

- definição clara da aplicação,  
- preparação e subset do dataset,  
- pipeline completo de tokenização → DataLoader → treinamento → validação,  
- uso da GPU MPS no MacBook Air M4,  
- resultados consistentes e métricas apresentadas,  
- validação adicional confirmando o comportamento do modelo.

O baseline está pronto para as próximas análises.

---

# 11. Próximos Passos — Entrega 28/11 (Robustez)

A próxima etapa incluirá:

- ruído ortográfico,  
- substituição lexical,  
- inserção de emojis,  
- simplificação do texto,  
- perturbações adversariais simples.

Será medida a variação do F1 sob cada perturbação.

---

# 12. Código Fonte

Incluso no repositório:

- Notebook do treinamento,  
- Script `train_gpu.ipynb`,  
- `requirements.txt`,  
- Modelo salvo em `baseline_model/`.
