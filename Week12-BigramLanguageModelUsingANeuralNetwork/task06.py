import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
EPOCHS = 100
LR = 7  # it explodes with 70


def get_vocabulary(names):
    vocabulary = set()

    for name in names:
        vocabulary.update(name)

    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, ".")

    return vocabulary


def one_hot_encode(xs, vocab_size):
    return F.one_hot(xs, num_classes=vocab_size).float()


def create_training_set(names, vocabulary):
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    xss, yss = [], []
    for name in names:
        xs, ys = create_training_set_one_name(name, char_to_idx)
        xss.append(xs)
        yss.append(ys)

    xss = torch.cat(xss, dim=0)
    yss = torch.cat(yss, dim=0)

    return xss, yss


def create_training_set_one_name(name, char_to_idx):
    full_sequence = ["."] + list(name) + ["."]

    xs, ys = [], []
    for prev, next in zip(full_sequence, full_sequence[1:]):
        i = char_to_idx[prev]
        j = char_to_idx[next]

        xs.append(i)
        ys.append(j)

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    return xs, ys


def train_bigram_model(xs, ys, vocab_size, epochs, lr):
    g = torch.Generator().manual_seed(2147483647)

    w1 = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    w2 = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    xs_oh = one_hot_encode(xs, vocab_size)

    losses = []
    for epoch in range(epochs):
        hidden = xs_oh @ w1
        logits = hidden @ w2

        probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(probs, ys)
        losses.append(loss.item())

        loss.backward()

        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()

        print(f"Epoch [{epoch+1}/{epochs}]: {loss.item()}")

    return w1, w2, losses


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model loss")
    plt.show()


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    vocabulary = get_vocabulary(names)
    vocab_size = len(vocabulary)

    xs, ys = create_training_set(names, vocabulary)

    print(f"Training with {len(xs)} examples.")

    w1, w2, losses = train_bigram_model(xs,
                                        ys,
                                        vocab_size,
                                        epochs=EPOCHS,
                                        lr=LR)
    plot_losses(losses)


if __name__ == '__main__':
    main()
