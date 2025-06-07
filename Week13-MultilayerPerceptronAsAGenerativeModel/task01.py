import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

PATH = os.path.join("..", "DATA", "names.txt")


def get_vocabulary(names):
    vocabulary = set()

    for name in names:
        vocabulary.update(name)

    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, ".")

    return vocabulary


def main():
    with open(PATH, 'r') as f:
        names = f.read().split()

    vocabulary = get_vocabulary(names)
    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}

    print(f"First 8 words: {names[:8]}")
    print(f"Total number of words: {len(names)}")
    print(f"Integer to character map: {idx_to_char}")


if __name__ == '__main__':
    main()
