# Exemplos de MLP (Redes Neurais Feedforward)

Este diretório contém dois exemplos simples de redes neurais feedforward (MLP) implementados com PyTorch:

- [`classification/train.py`](classification/train.py): classificação multiclasse.
- [`regression/train.py`](regression/train.py): regressão.

## Como executar

A partir da raiz do repositório, com o ambiente virtual ativado e as dependências de `requirements.txt` instaladas:

```bash
python mlp/classification/train.py
python mlp/regression/train.py
```

Cada script é autocontido: gera os dados, treina o modelo, avalia no conjunto de teste e salva a curva de aprendizado em `outputs/learning_curve.png` dentro da sua respectiva pasta.

## Exemplo de classificação

Dados: dataset sintético gerado com `sklearn.datasets.make_classification` (20 atributos, 3 classes), dividido em treino (70%), validação (15%) e teste (15%), com padronização (`StandardScaler`) ajustada apenas no conjunto de treino.

Arquitetura (`MLPClassifier`):

| Camada  | Transformação      | Ativação |
|---------|---------------------|----------|
| Entrada | 20 features          | –        |
| Oculta 1| `Linear(20, 64)`     | ReLU     |
| Oculta 2| `Linear(64, 32)`     | ReLU     |
| Saída   | `Linear(32, 3)`      | Nenhuma (logits) |

- **Função de perda**: `CrossEntropyLoss` (aplica internamente `LogSoftmax`, por isso a camada de saída não tem ativação).
- **Otimizador**: `Adam` (lr = 1e-3).
- **Métricas**: perda e acurácia por época (treino e validação); acurácia, precisão, recall e f1-score no conjunto de teste (`classification_report`).

## Exemplo de regressão

Dados: dataset sintético não linear ($y = \sin(x_0) + 0.5x_1^2 - x_2 + \text{ruído}$), dividido da mesma forma (70/15/15). Tanto as features quanto o alvo são padronizados (`StandardScaler`) com base no conjunto de treino; a padronização do alvo é revertida antes do cálculo das métricas finais.

Arquitetura (`MLPRegressor`):

| Camada  | Transformação      | Ativação |
|---------|---------------------|----------|
| Entrada | 3 features           | –        |
| Oculta 1| `Linear(3, 64)`      | ReLU     |
| Oculta 2| `Linear(64, 32)`     | ReLU     |
| Saída   | `Linear(32, 1)`      | Nenhuma (saída contínua) |

- **Função de perda**: `MSELoss` (erro quadrático médio).
- **Otimizador**: `Adam` (lr = 1e-3).
- **Métricas**: perda (MSE) e R² por época (treino e validação) — como não existe "acurácia" em problemas de regressão, o R² é usado como métrica equivalente, pois mede a fração da variância explicada pelo modelo. No conjunto de teste, também são reportados MSE, MAE e R².

## Saídas geradas

Ao final da execução, cada exemplo:
1. Imprime no console o progresso do treinamento (perda/métrica a cada 10 épocas).
2. Salva a curva de aprendizado (perda e métrica principal, para treino e validação) em `outputs/learning_curve.png`.
3. Imprime as métricas de avaliação no conjunto de teste.
