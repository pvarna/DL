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


def single_neuron(xs_oh, vocab_size):
    torch.manual_seed(SEED)

    w = torch.randn((vocab_size, 1))

    logits = xs_oh @ w
    return logits


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    vocabulary = get_vocabulary(names)
    vocab_size = len(vocabulary)

    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    emma_xs, _ = create_training_set_one_name("emma", char_to_idx)
    emma_xs_oh = one_hot_encode(emma_xs, vocab_size)

    emma_logits = single_neuron(emma_xs_oh, vocab_size)

    print(emma_logits)
    print(emma_logits.shape)


if __name__ == '__main__':
    main()
