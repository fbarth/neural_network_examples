# Exemplo de rede convolucional (CNN)

Este diretório contém um exemplo de classificação de imagens usando PyTorch.

## Como executar

A partir da raiz do repositório, com o ambiente virtual ativado e as dependências de `requirements.txt` instaladas:

```bash
python -m convolutional.train
```

O conjunto de dígitos manuscritos (`load_digits`) é fornecido pelo scikit-learn, então a execução não baixa dados. O script separa as imagens em treino (70%), validação (15%) e teste (15%), treina o modelo, mostra as métricas no conjunto de teste e salva as curvas em `convolutional/outputs/learning_curve.png`.

## Dados e arquitetura

Cada imagem em tons de cinza tem tamanho 8 × 8 pixels e os valores dos pixels são normalizados para o intervalo [0, 1]. A arquitetura `CNNClassifier` é:

| Camada | Transformação | Ativação |
|--------|---------------|----------|
| Entrada | 1 canal, 8 × 8 pixels | – |
| Convolucional 1 | `Conv2d(1, 16, kernel_size=3, padding=1)` | ReLU |
| Pooling 1 | `MaxPool2d(kernel_size=2)` | – |
| Convolucional 2 | `Conv2d(16, 32, kernel_size=3, padding=1)` | ReLU |
| Pooling 2 | `MaxPool2d(kernel_size=2)` | – |
| Densa | `Linear(128, 64)` | ReLU |
| Saída | `Linear(64, 10)` | Nenhuma (logits) |

- **Função de perda**: `CrossEntropyLoss` (recebe os logits diretamente).
- **Otimizador**: `Adam`, com taxa de aprendizado `1e-3`.
- **Métricas**: perda e acurácia por época para treino e validação; acurácia, precisão, recall e F1-score no conjunto de teste.

O gráfico de aprendizado mostra a perda e a acurácia ao longo das épocas, cada uma comparando os conjuntos de treino e validação.
