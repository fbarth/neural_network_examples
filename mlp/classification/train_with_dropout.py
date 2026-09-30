"""Exemplo de MLP (feedforward) para classificação multiclasse com PyTorch,
incluindo camadas de Dropout para reduzir overfitting.

Arquitetura:
    Entrada (20 features) -> Linear(20, 64) -> ReLU -> Dropout(0.3)
                           -> Linear(64, 32) -> ReLU -> Dropout(0.3)
                           -> Linear(32, 3)  -> logits (3 classes)

Função de perda: CrossEntropyLoss (combina LogSoftmax + NLLLoss).
Otimizador: Adam.
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_EPOCHS = 500 # colocando um número maior de épocas visto que o dropout pode 
# retardar a convergência
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


class MLPClassifier(nn.Module):
    """MLP com duas camadas ocultas, ativação ReLU e Dropout para regularização."""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(32, n_classes),  # logits, sem ativação (usados diretamente pela CrossEntropyLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def prepare_data():
    """Gera um dataset sintético de classificação e o divide em treino/validação/teste."""
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=SEED,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    to_tensor = lambda arr, dtype: torch.tensor(arr, dtype=dtype)
    train_ds = TensorDataset(to_tensor(X_train, torch.float32), to_tensor(y_train, torch.long))
    val_ds = TensorDataset(to_tensor(X_val, torch.float32), to_tensor(y_val, torch.long))
    test_ds = TensorDataset(to_tensor(X_test, torch.float32), to_tensor(y_test, torch.long))

    return train_ds, val_ds, test_ds


def run_epoch(model, loader, criterion, optimizer=None):
    """Executa uma época de treino (se optimizer for passado) ou avaliação."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X_batch, y_batch in loader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += X_batch.size(0)

    return total_loss / total_samples, total_correct / total_samples


def plot_learning_curves(history, output_path):
    """Plota a evolução da perda e da acurácia para treino e validação."""
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
    print(f"Curva de aprendizado salva em: {output_path}")


def main():
    torch.manual_seed(SEED)

    train_ds, val_ds, test_ds = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = train_ds.tensors[0].shape[1]
    n_classes = int(train_ds.tensors[1].max().item() + 1)
    model = MLPClassifier(n_features, n_classes)

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

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Época {epoch:3d}/{N_EPOCHS} | "
                f"treino: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f}"
            )

    plot_learning_curves(history, os.path.join(OUTPUT_DIR, "learning_curve_dropout.png"))

    model.eval()
    with torch.no_grad():
        X_test, y_test = test_ds.tensors
        y_pred = model(X_test).argmax(dim=1)

    print("\nAvaliação no conjunto de teste:")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
