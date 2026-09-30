# Especificações gerais do agente

Veja o objetivo deste projeto no arquivo README.md. 

Você deverá criar exemplos de redes neurais utilizando diferentes arquiteturas, seguindo a organização apresentada no README.md. Certifique-se de incluir exemplos de redes feedforward (MLP), convolucionais e recorrentes, conforme os diretórios indicados.

Todos os exemplos devem usar o framework PyTorch como base.

Além disso, você deve garantir que cada exemplo esteja devidamente documentado, incluindo instruções de execução e explicações sobre a arquitetura utilizada.

Para a validação de cada modelo, você deve incluir a curva de aprendizado, mostrando a evolução da função de perda e da acurácia ao longo das épocas de treinamento. Tanto para o conjunto de treinamento quanto para o conjunto de validação.

Você deve começar com o exemplo sobre MLP, localizado no diretório `mlp`. Depois de validado este exemplo então podemos desenvolver os exemplos das outras arquiteturas, seguindo a mesma lógica de documentação, execução e validação.

## Requisitos para o exemplo sobre MLP

Os exemplos apresentados devem ser simples: um para classificação e outro para regressão.
Cada exemplo deve incluir:
- Definição da arquitetura da rede neural.
- Preparação dos dados de entrada.
- Treinamento do modelo.
- Avaliação do modelo com métricas apropriadas.
- Visualização da curva de aprendizado.
  
Deve estar claro no código a função de ativação utilizada em cada camada da rede neural, a função de perda utilizada e o otimizador escolhido para o treinamento.