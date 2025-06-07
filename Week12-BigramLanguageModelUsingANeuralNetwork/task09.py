import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
EPOCHS = 100
LR = 5  # it explodes with 7
LAMBDA_REG = 0.01


def get_vocabulary(names):
    vocabulary = set()

    for name in names:
        vocabulary.update(name)

    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, ".")

    return vocabulary


def one_hot_encode_trigram(xs, vocab_size):
    x1 = F.one_hot(xs[:, 0], num_classes=vocab_size)
    x2 = F.one_hot(xs[:, 1], num_classes=vocab_size)
    return torch.cat([x1, x2], dim=1).float()


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
    full_sequence = ['.', '.'] + list(name) + ['.']

    xs, ys = [], []
    for i in range(len(full_sequence) - 2):
        ch1 = full_sequence[i]
        ch2 = full_sequence[i + 1]
        ch3 = full_sequence[i + 2]

        idx1 = char_to_idx[ch1]
        idx2 = char_to_idx[ch2]
        idx3 = char_to_idx[ch3]

        xs.append((idx1, idx2))
        ys.append(idx3)

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    return xs, ys


def train_trigram_model(xs, ys, vocab_size, epochs, lr, lambda_reg):
    g = torch.Generator().manual_seed(2147483647)

    w1 = torch.randn((2 * vocab_size, vocab_size),
                     generator=g,
                     requires_grad=True)
    w2 = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    xs_oh = one_hot_encode_trigram(xs, vocab_size)

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

        print(f"Epoch [{epoch+1}/{epochs}]: {loss.item()}")

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
        ctx = [char_to_idx['.'], char_to_idx['.']]

        while True:
            x1 = F.one_hot(torch.tensor([ctx[0]]), num_classes=vocab_size)
            x2 = F.one_hot(torch.tensor([ctx[1]]), num_classes=vocab_size)
            x = torch.cat([x1, x2], dim=1).float()

            hidden = x @ w1
            logits = hidden @ w2
            probs = F.softmax(logits, dim=1)

            idx = torch.multinomial(probs, num_samples=1).item()
            char = idx_to_char[idx]

            if char == '.':
                break

            word += char
            ctx = [ctx[1], idx]

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

    w1, w2, losses = train_trigram_model(xs,
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
    print(f"Final trigram model loss: {losses[-1]}")

    # It did improve - the loss is now ~2.42 (which is even lower than the bigram model via counting)


if __name__ == '__main__':
    main()
