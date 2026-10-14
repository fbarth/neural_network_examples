# Exemplos de redes recorrentes

Estes exemplos estimam a cotação diária de venda do dólar dos Estados Unidos em reais (USD/BRL) com duas arquiteturas: uma RNN simples e uma LSTM. Ambos usam os mesmos dados, janelas, divisões temporais e parâmetros para permitir comparação.

## Como executar

Na raiz do repositório, com as dependências de `requirements.txt` instaladas:

```bash
python -m recurrent.train_rnn
python -m recurrent.train_lstm
```

Na primeira execução, o script baixa as 2.500 observações mais recentes da série diária SGS 1 do [Banco Central do Brasil](https://dadosabertos.bcb.gov.br/dataset/1-taxa-de-cambio---livre---dolar-americano-venda---diario) para `recurrent/data/usd_brl_daily.json`. Como a API limita consultas de "últimos valores" a 20 registros, o carregador consulta intervalos de um ano e junta as respostas antes de selecionar os dados mais recentes. As execuções seguintes reutilizam o arquivo em cache; é necessária conexão com a internet para o primeiro download. A série mede a taxa de câmbio em reais por dólar americano.

Cada execução salva em `recurrent/outputs/` uma curva de aprendizado com MSE e R² para treino e validação, além de um gráfico comparando previsões e valores reais no período de teste.

## Dados e preparação

O modelo recebe as 30 cotações diárias anteriores e estima a próxima observação disponível (dias sem cotação, como fins de semana e feriados, não são incluídos pelo Banco Central). A série é dividida cronologicamente: 70% para treino, 15% para validação e 15% para teste. A normalização usa somente a média e o desvio padrão da parte de treino; as métricas finais e o gráfico de previsões voltam à escala original, em R$/US$.

## Arquiteturas e treinamento

| Exemplo | Camada recorrente | Camada de saída | Ativação |
|---------|------------------|-----------------|----------|
| RNN | `nn.RNN(1, 32, batch_first=True, nonlinearity="tanh")` | `nn.Linear(32, 1)` | `tanh` interna da RNN; saída linear |
| LSTM | `nn.LSTM(1, 32, batch_first=True)` | `nn.Linear(32, 1)` | Portas sigmoid e estado candidato tanh internos da LSTM; saída linear |

- **Função de perda**: `nn.MSELoss`.
- **Otimizador**: `Adam`, taxa de aprendizado `1e-3`.
- **Métricas**: MSE, MAE e R² no conjunto de teste. Para regressão, R² é apresentado em vez de acurácia.
- **Curvas**: MSE e R² por época, para treino e validação.
