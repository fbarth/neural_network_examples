"""Previsão da cotação diária USD/BRL com uma RNN simples."""

from recurrent.common import train_and_evaluate


if __name__ == "__main__":
    train_and_evaluate("rnn")
