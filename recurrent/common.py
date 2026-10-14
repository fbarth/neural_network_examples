"""Utilitários compartilhados pelos exemplos RNN e LSTM de séries temporais."""

import json
import os
from datetime import date, datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
WINDOW_SIZE = 30
N_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
DATA_PATH = os.path.join(DATA_DIR, "usd_brl_daily.json")
N_OBSERVATIONS = 2500


class ExchangeRateForecaster(nn.Module):
    """Prevê a próxima cotação com uma camada recorrente RNN ou LSTM."""

    def __init__(self, cell_type: str, hidden_size: int = 32):
        super().__init__()
        if cell_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=1, hidden_size=hidden_size, batch_first=True, nonlinearity="tanh"
            )
        elif cell_type == "lstm":
            self.recurrent = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        else:
            raise ValueError("cell_type deve ser 'rnn' ou 'lstm'.")

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(sequences)
        return self.output(outputs[:, -1, :]).squeeze(-1)


def load_exchange_rate_series():
    """Carrega o câmbio diário USD/BRL do SGS/BCB, baixando e armazenando o JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = None
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as source:
                cached = json.load(source)
            if isinstance(cached, list) and len(cached) >= WINDOW_SIZE + 20:
                rows = cached
        except (OSError, json.JSONDecodeError):
            pass

    if rows is None:
        # O endpoint /ultimos aceita no máximo 20 registros. Consultamos anos
        # individuais pela API de intervalo e depois mantemos as observações recentes.
        latest_year = date.today().year
        first_year = latest_year - 11
        rows = []
        request_headers = {"User-Agent": "neural-networks-examples/1.0"}
        try:
            for year in range(first_year, latest_year + 1):
                query = urlencode(
                    {
                        "formato": "json",
                        "dataInicial": f"01/01/{year}",
                        "dataFinal": f"31/12/{year}",
                    }
                )
                request = Request(f"{DATA_URL}?{query}", headers=request_headers)
                with urlopen(request, timeout=30) as response:
                    year_rows = json.loads(response.read().decode("utf-8"))
                if isinstance(year_rows, dict) and "erro" in year_rows:
                    raise RuntimeError(year_rows["erro"].get("detail", "erro da API SGS"))
                if not isinstance(year_rows, list):
                    raise RuntimeError(f"Resposta inesperada da API SGS para {year}.")
                rows.extend(year_rows)

            # Só substitui o cache depois que todos os intervalos foram obtidos.
            temp_path = f"{DATA_PATH}.tmp"
            with open(temp_path, "w", encoding="utf-8") as output:
                json.dump(rows, output)
            os.replace(temp_path, DATA_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível baixar os dados do Banco Central pela API SGS ({DATA_URL}). "
                "Verifique a conexão e tente novamente."
            ) from exc

    dates, values = [], []
    for row in rows:
        try:
            observation_date = datetime.strptime(row["data"], "%d/%m/%Y")
            value = float(str(row["valor"]).replace(",", "."))
        except (KeyError, TypeError, ValueError):
            continue
        dates.append(observation_date)
        values.append(value)

    if len(values) < WINDOW_SIZE + 20:
        raise ValueError(f"O arquivo {DATA_PATH} não contém observações suficientes.")
    ordered = sorted(zip(dates, values), key=lambda item: item[0])[-N_OBSERVATIONS:]
    dates, values = zip(*ordered)
    return list(dates), np.asarray(values, dtype=np.float32)


def make_sequences(values: np.ndarray, window_size: int):
    """Cria janelas ordenadas; cada alvo é a observação seguinte à janela."""
    X = np.stack([values[i : i + window_size] for i in range(len(values) - window_size)])
    y = values[window_size:]
    return X[:, :, None], y


def prepare_data():
    """Divide cronologicamente em treino/validação/teste e normaliza sem vazamento."""
    dates, values = load_exchange_rate_series()
    n = len(values)
    train_end, val_end = int(n * 0.70), int(n * 0.85)

    mean = float(values[:train_end].mean())
    scale = float(values[:train_end].std()) or 1.0
    scaled = (values - mean) / scale
    X, y = make_sequences(scaled, WINDOW_SIZE)
    target_indices = np.arange(WINDOW_SIZE, n)

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    def dataset(mask):
        return TensorDataset(
            torch.tensor(X[mask], dtype=torch.float32),
            torch.tensor(y[mask], dtype=torch.float32),
        )

    return (
        dataset(train_mask),
        dataset(val_mask),
        dataset(test_mask),
        dates,
        mean,
        scale,
        target_indices[test_mask],
    )


def run_epoch(model, loader, criterion, optimizer=None):
    """Executa treino ou avaliação e retorna MSE e R² no espaço normalizado."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, total_samples = 0.0, 0
    predictions, targets = [], []

    for sequences, actual in loader:
        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            predicted = model(sequences)
            loss = criterion(predicted, actual)
        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        total_samples += sequences.size(0)
        predictions.append(predicted.detach())
        targets.append(actual)

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / total_samples, r2_score(targets, predictions)


def plot_learning_curves(history, output_path, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))
    ax_loss.plot(epochs, history["train_loss"], label="Treino")
    ax_loss.plot(epochs, history["val_loss"], label="Validação")
    ax_loss.set(title="Curva de perda (MSE)", xlabel="Época", ylabel="Perda")
    ax_loss.legend()
    ax_r2.plot(epochs, history["train_r2"], label="Treino")
    ax_r2.plot(epochs, history["val_r2"], label="Validação")
    ax_r2.set(title="Curva de R²", xlabel="Época", ylabel="R²")
    ax_r2.legend()
    fig.suptitle(f"{model_name} — câmbio diário USD/BRL")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Curvas de aprendizado salvas em: {output_path}")


def plot_predictions(dates, target_indices, actual, predicted, output_path, model_name):
    fig, ax = plt.subplots(figsize=(12, 5))
    test_dates = [dates[i] for i in target_indices]
    ax.plot(test_dates, actual, label="Valor real", linewidth=1.2)
    ax.plot(test_dates, predicted, label="Previsão", linewidth=1.2)
    ax.set(
        title=f"{model_name} — previsões no conjunto de teste",
        xlabel="Data",
        ylabel="Cotação de venda (R$/US$)",
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Gráfico de previsões salvo em: {output_path}")


def train_and_evaluate(cell_type: str):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_ds, val_ds, test_ds, dates, mean, scale, test_indices = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ExchangeRateForecaster(cell_type=cell_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {key: [] for key in ("train_loss", "train_r2", "val_loss", "val_r2")}

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for sequences, actual in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(sequences), actual)
            loss.backward()
            optimizer.step()

        train_loss, train_r2 = run_epoch(model, train_eval_loader, criterion)
        val_loss, val_r2 = run_epoch(model, val_loader, criterion)
        for key, value in zip(history, (train_loss, train_r2, val_loss, val_r2)):
            history[key].append(value)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Época {epoch:2d}/{N_EPOCHS} | "
                f"treino: MSE={train_loss:.4f} R²={train_r2:.4f} | "
                f"validação: MSE={val_loss:.4f} R²={val_r2:.4f}"
            )

    model_name = "RNN" if cell_type == "rnn" else "LSTM"
    plot_learning_curves(
        history, os.path.join(OUTPUT_DIR, f"{cell_type}_learning_curve.png"), model_name
    )

    X_test, y_test_scaled = test_ds.tensors
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(X_test).numpy()
    actual = y_test_scaled.numpy() * scale + mean
    predicted = predicted_scaled * scale + mean

    print(f"\nAvaliação {model_name} no conjunto de teste:")
    print(f"MSE: {mean_squared_error(actual, predicted):.4f}")
    print(f"MAE: {mean_absolute_error(actual, predicted):.4f}")
    print(f"R²:  {r2_score(actual, predicted):.4f}")
    plot_predictions(
        dates,
        test_indices,
        actual,
        predicted,
        os.path.join(OUTPUT_DIR, f"{cell_type}_predictions.png"),
        model_name,
    )
