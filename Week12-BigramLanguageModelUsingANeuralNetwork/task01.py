import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def get_vocabulary(names):
    vocabulary = set()

    for name in names:
        vocabulary.update(name)

    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, ".")

    return vocabulary


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

    xs = torch.tensor(xs, dtype=torch.int32)
    ys = torch.tensor(ys, dtype=torch.int32)

    return xs, ys


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    vocabulary = get_vocabulary(names)

    print(create_training_set(names, vocabulary))

    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    xs, ys = create_training_set_one_name("emma", char_to_idx)
    print(f"{xs=}")
    print(f"{ys=}")


if __name__ == '__main__':
    main()
