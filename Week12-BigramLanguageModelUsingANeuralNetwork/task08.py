import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
EPOCHS = 100
LR = 7  # it explodes with 70
LAMBDA_REG = 0.01


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


def train_bigram_model(xs, ys, vocab_size, epochs, lr, lambda_reg):
    g = torch.Generator().manual_seed(2147483647)

    w1 = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    w2 = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    xs_oh = one_hot_encode(xs, vocab_size)

    losses = []
    for epoch in range(epochs):
        hidden = xs_oh @ w1
        logits = hidden @ w2

        probs = F.log_softmax(logits, dim=1)
        reg_loss = lambda_reg * ((w1**2).mean() + (w2**2).mean())
        loss = F.nll_loss(probs, ys) + reg_loss
        losses.append(loss.item())

        loss.backward()

        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()

    return w1, w2, losses


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model loss")
    plt.show()


def generate_words(w1, w2, vocab_size, idx_to_char, char_to_idx, num_words):
    words = []

    for _ in range(num_words):
        word = ''
        idx = char_to_idx['.']

        while True:
            x = torch.zeros((1, vocab_size))
            x[0, idx] = 1.0

            hidden = x @ w1
            logits = hidden @ w2
            probs = F.softmax(logits, dim=1)

            idx = torch.multinomial(probs, num_samples=1).item()
            char = idx_to_char[idx]

            if char == '.':
                break

            word += char

        words.append(word)

    return words


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    vocabulary = get_vocabulary(names)
    vocab_size = len(vocabulary)

    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}

    xs, ys = create_training_set(names, vocabulary)

    w1, w2, losses = train_bigram_model(xs,
                                        ys,
                                        vocab_size,
                                        epochs=EPOCHS,
                                        lr=LR,
                                        lambda_reg=LAMBDA_REG)

    words = generate_words(w1,
                           w2,
                           vocab_size,
                           idx_to_char,
                           char_to_idx,
                           num_words=10)
    print(words)

    # The generated words are similar to the ones generated via the model using counting


if __name__ == '__main__':
    main()
