"""Exemplo de MLP (feedforward) para regressão com PyTorch.

Arquitetura:
    Entrada (3 features) -> Linear(3, 64) -> ReLU
                          -> Linear(64, 32) -> ReLU
                          -> Linear(32, 1)  -> saída contínua (sem ativação)

Função de perda: MSELoss (erro quadrático médio).
Otimizador: Adam.

Como métrica de "acurácia" (não existe acurácia em regressão), usamos o
coeficiente de determinação R², que mede a fração da variância explicada
pelo modelo (quanto mais próximo de 1, melhor).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


class MLPRegressor(nn.Module):
    """MLP com duas camadas ocultas e ativação ReLU para regressão."""

    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # saída contínua, sem ativação
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def prepare_data():
    """Gera um dataset sintético não linear e o divide em treino/validação/teste."""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-3, 3, size=(2000, 3)).astype(np.float32)
    noise = rng.normal(0, 0.5, size=2000).astype(np.float32)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + noise

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    x_scaler = StandardScaler().fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    # Padroniza também o alvo para estabilizar o treinamento; a transformação
    # é revertida antes do cálculo das métricas finais.
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train_s))
    val_ds = TensorDataset(to_tensor(X_val), to_tensor(y_val_s))
    test_ds = TensorDataset(to_tensor(X_test), to_tensor(y_test))

    return train_ds, val_ds, test_ds, y_scaler


def run_epoch(model, loader, criterion, optimizer=None):
    """Executa uma época de treino (se optimizer for passado) ou avaliação.

    Retorna a perda média (MSE) e o R² da época.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
        all_preds.append(preds.detach())
        all_targets.append(y_batch)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    r2 = r2_score(all_targets, all_preds)

    return total_loss / total_samples, r2


def plot_learning_curves(history, output_path):
    """Plota a evolução da perda (MSE) e do R² para treino e validação."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Treino")
    ax_loss.plot(epochs, history["val_loss"], label="Validação")
    ax_loss.set_title("Curva de perda (MSE)")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Perda")
    ax_loss.legend()

    ax_r2.plot(epochs, history["train_r2"], label="Treino")
    ax_r2.plot(epochs, history["val_r2"], label="Validação")
    ax_r2.set_title("Curva de R² (equivalente à acurácia em regressão)")
    ax_r2.set_xlabel("Época")
    ax_r2.set_ylabel("R²")
    ax_r2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"Curva de aprendizado salva em: {output_path}")


def main():
    torch.manual_seed(SEED)

    train_ds, val_ds, test_ds, y_scaler = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = train_ds.tensors[0].shape[1]
    model = MLPRegressor(n_features)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_r2": [], "val_loss": [], "val_r2": []}
    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_r2 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_r2 = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_r2"].append(train_r2)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Época {epoch:3d}/{N_EPOCHS} | "
                f"treino: loss={train_loss:.4f} r2={train_r2:.4f} | "
                f"val: loss={val_loss:.4f} r2={val_r2:.4f}"
            )

    plot_learning_curves(history, os.path.join(OUTPUT_DIR, "learning_curve.png"))

    model.eval()
    with torch.no_grad():
        X_test, y_test = test_ds.tensors
        y_pred_scaled = model(X_test).numpy()

    # Reverte a padronização do alvo para obter as métricas na escala original.
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test = y_test.numpy()

    print("\nAvaliação no conjunto de teste:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R²:  {r2_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
