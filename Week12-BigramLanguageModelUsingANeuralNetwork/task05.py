import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


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


def two_layer_network(xs_oh, vocab_size):
    torch.manual_seed(SEED)

    w1 = torch.randn((vocab_size, vocab_size))  # input 27 -> hidden 27
    hidden = xs_oh @ w1

    w2 = torch.randn((vocab_size, vocab_size))  # hidden 27 -> output 27
    logits = hidden @ w2

    counts = torch.exp(logits)
    probs = counts / counts.sum(dim=1, keepdim=True)

    return probs


def trace_bigram_predictions(xs, ys, probs, idx_to_char):
    total_nll = 0.0

    for i in range(len(xs)):
        x_idx = xs[i].item()
        y_idx = ys[i].item()
        prob_vector = probs[i]
        predicted_prob = prob_vector[y_idx].item()
        log_likelihood = torch.log(torch.tensor(predicted_prob))
        nll = -log_likelihood.item()
        total_nll += nll

        print("------")
        print(
            f"bigram example {i + 1}: {idx_to_char[x_idx]}{idx_to_char[y_idx]} (indexes {x_idx},{y_idx})"
        )
        print(f"input to the neural net: {x_idx}")
        print(f"output probabilities from the neural net: {prob_vector}")
        print(f"label (actual next character): {y_idx}")
        print(
            f"probability assigned by the neural net to the label: {predicted_prob}"
        )
        print(f"log likelihood: {log_likelihood.item()}")
        print(f"negative log likelihood: {nll}")

    average_nll = total_nll / len(xs)
    print("======")
    print(f"average negative log likelihood, i.e. loss = {average_nll}")


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    vocabulary = get_vocabulary(names)
    vocab_size = len(vocabulary)

    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}

    emma_xs, emma_ys = create_training_set_one_name("emma", char_to_idx)
    emma_xs_oh = one_hot_encode(emma_xs, vocab_size)

    emma_probs = two_layer_network(emma_xs_oh, vocab_size)

    trace_bigram_predictions(emma_xs, emma_ys, emma_probs, idx_to_char)


if __name__ == '__main__':
    main()
