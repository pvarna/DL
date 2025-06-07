import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

SEED = 42
EPOCHS = 100
LR = 5
LAMBDA_REG = 0.01
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1


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


def create_training_set_trigram(names, vocabulary):
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    xss, yss = [], []

    for name in names:
        xs, ys = create_training_set_one_name_trigram(name, char_to_idx)
        xss.append(xs)
        yss.append(ys)

    xs_tensor = torch.cat(xss, dim=0)
    ys_tensor = torch.cat(yss, dim=0)

    return xs_tensor, ys_tensor


def create_training_set_one_name_trigram(name, char_to_idx):
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


def train_trigram_model(xs_train,
                        ys_train,
                        xs_val,
                        ys_val,
                        vocab_size,
                        epochs,
                        lr,
                        lambda_reg,
                        print_loss=False):
    g = torch.Generator().manual_seed(2147483647)

    w1 = torch.randn((2 * vocab_size, vocab_size),
                     generator=g,
                     requires_grad=True)
    w2 = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    xs_train_oh = one_hot_encode_trigram(xs_train, vocab_size)
    xs_val_oh = one_hot_encode_trigram(xs_val, vocab_size)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        hidden_train = xs_train_oh @ w1
        logits_train = hidden_train @ w2

        train_probs = F.log_softmax(logits_train, dim=1)
        reg_loss = lambda_reg * ((w1**2).mean() + (w2**2).mean())
        train_loss = F.nll_loss(train_probs, ys_train) + reg_loss
        train_losses.append(train_loss.item())

        train_loss.backward()

        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()

        with torch.no_grad():
            hidden_val = xs_val_oh @ w1
            logits_val = hidden_val @ w2

            val_probs = F.log_softmax(logits_val, dim=1)
            val_loss = F.nll_loss(val_probs, ys_val)
            val_losses.append(val_loss.item())

        if print_loss:
            print(
                f"Epoch [{epoch+1}/{epochs}]: train_loss={train_loss.item()}, val_loss={val_loss.item()}"
            )

    return w1, w2, train_losses, val_losses


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation loss")
    plt.legend()
    plt.show()


def find_best_lambda(xs_train, ys_train, xs_val, ys_val, vocab_size, epochs,
                     lr):
    best_val_loss = torch.inf
    best_lambda = None
    results = {}

    for lambda_reg in [0.0, 0.001, 0.01, 0.1]:
        _, _, _, val_losses = train_trigram_model(xs_train, ys_train, xs_val,
                                                  ys_val, vocab_size, epochs,
                                                  lr, lambda_reg)

        results[lambda_reg] = val_losses[-1]
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_lambda = lambda_reg

    return best_lambda


def generate_words_trigram(w1, w2, vocab_size, idx_to_char, char_to_idx,
                           num_words):
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


def split_dataset(xs, ys):
    total_size = len(xs)
    train_size = int(total_size * TRAIN_FRAC)
    val_size = int(total_size * VAL_FRAC)
    test_size = total_size - train_size - val_size

    g = torch.Generator().manual_seed(2147483647)
    dataset = list(zip(xs, ys))
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=g)

    def unzip(paired):
        x, y = zip(*paired)
        return torch.stack(x), torch.tensor(y)

    return unzip(train_set), unzip(val_set), unzip(test_set)


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    vocabulary = get_vocabulary(names)
    vocab_size = len(vocabulary)

    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}

    xs, ys = create_training_set_trigram(names, vocabulary)

    (xs_train, ys_train), (xs_val, ys_val), (xs_test,
                                             ys_test) = split_dataset(xs, ys)

    best_lambda = find_best_lambda(xs_train,
                                   ys_train,
                                   xs_val,
                                   ys_val,
                                   vocab_size,
                                   epochs=EPOCHS,
                                   lr=LR)
    print(f"Best lambda: {best_lambda}")

    w1, w2, train_losses, val_losses = train_trigram_model(
        xs_train,
        ys_train,
        xs_val,
        ys_val,
        vocab_size,
        epochs=EPOCHS,
        lr=LR,
        lambda_reg=best_lambda,
        print_loss=True)
    plot_losses(train_losses, val_losses)

    xs_test_oh = one_hot_encode_trigram(xs_test, vocab_size)
    with torch.no_grad():
        hidden_test = xs_test_oh @ w1
        logits_test = hidden_test @ w2
        test_probs = F.log_softmax(logits_test, dim=1)
        test_loss = F.nll_loss(test_probs, ys_test)

    print(f"Final test loss: {test_loss.item()}")

    generated_words = generate_words_trigram(w1,
                                             w2,
                                             vocab_size,
                                             idx_to_char,
                                             char_to_idx,
                                             num_words=10)
    print(f"Sample generated words: {generated_words}")

    print(
        f"The grid search for lambda further improved the decrease of the loss (it is ~ 2.40 now)"
    )


if __name__ == '__main__':
    main()
