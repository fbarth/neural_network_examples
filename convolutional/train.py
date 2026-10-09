"""Exemplo simples de CNN para classificação de imagens com PyTorch.

Arquitetura:
    Entrada (1 x 8 x 8) -> Conv2d(1, 16, 3) -> ReLU -> MaxPool2d(2)
                        -> Conv2d(16, 32, 3) -> ReLU -> MaxPool2d(2)
                        -> Flatten -> Linear(128, 64) -> ReLU
                        -> Linear(64, 10) -> logits

Função de perda: CrossEntropyLoss.
Otimizador: Adam.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


class CNNClassifier(nn.Module):
    """CNN com duas camadas convolucionais para classificar dígitos manuscritos."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def prepare_data():
    """Carrega imagens de dígitos e separa os conjuntos de treino, validação e teste."""
    images, labels = load_digits(return_X_y=True)
    images = (images.reshape(-1, 8, 8) / 16.0).astype("float32")[:, None, :, :]
    labels = np.asarray(labels, dtype=np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    to_tensor = lambda array, dtype: torch.tensor(array, dtype=dtype)
    train_ds = TensorDataset(to_tensor(X_train, torch.float32), to_tensor(y_train, torch.long))
    val_ds = TensorDataset(to_tensor(X_val, torch.float32), to_tensor(y_val, torch.long))
    test_ds = TensorDataset(to_tensor(X_test, torch.float32), to_tensor(y_test, torch.long))
    return train_ds, val_ds, test_ds


def run_epoch(model, loader, criterion, optimizer=None):
    """Executa uma época de treino ou avaliação e retorna perda e acurácia."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def plot_learning_curves(history, output_path):
    """Salva curvas de perda e acurácia dos conjuntos de treino e validação."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Treino")
    ax_loss.plot(epochs, history["val_loss"], label="Validação")
    ax_loss.set_title("Curva de perda (CrossEntropyLoss)")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Perda")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="Treino")
    ax_acc.plot(epochs, history["val_acc"], label="Validação")
    ax_acc.set_title("Curva de acurácia")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Acurácia")
    ax_acc.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Curva de aprendizado salva em: {output_path}")


def main():
    torch.manual_seed(SEED)

    train_ds, val_ds, test_ds = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNClassifier(n_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Época {epoch:2d}/{N_EPOCHS} | "
                f"treino: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f}"
            )

    plot_learning_curves(history, os.path.join(OUTPUT_DIR, "learning_curve.png"))

    model.eval()
    X_test, y_test = test_ds.tensors
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1)

    print("\nAvaliação no conjunto de teste:")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
