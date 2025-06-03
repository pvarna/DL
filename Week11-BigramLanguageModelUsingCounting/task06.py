import torch
import matplotlib.pyplot as plt


def generate_bigrams(name):
    bigrams = []

    bigrams.append((".", name[0]))

    for i in range(len(name) - 1):
        bigrams.append((name[i], name[i + 1]))

    bigrams.append((name[-1], "."))

    return bigrams


def get_vocabulary(names):
    vocabulary = set()

    for name in names:
        vocabulary.update(name)

    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, ".")

    return vocabulary


def create_bigram_matrix(names):
    vocabulary = get_vocabulary(names)
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}

    N = len(vocabulary)
    bigram_matrix = torch.zeros((N, N), dtype=torch.int32)

    for name in names:
        bigrams = generate_bigrams(name)
        for prev, next in bigrams:
            i = char_to_idx[prev]
            j = char_to_idx[next]
            bigram_matrix[i, j] += 1

    return bigram_matrix, vocabulary


def plot_bigram_heatmap(bigram_matrix, vocabulary):
    plt.figure(figsize=(16, 16))
    plt.imshow(bigram_matrix, cmap="Blues")

    for i in range(len(vocabulary)):
        for j in range(len(vocabulary)):
            count = bigram_matrix[i, j].item()
            bigram = vocabulary[i] + vocabulary[j]
            plt.text(j,
                     i,
                     f"{bigram}\n{count}",
                     ha="center",
                     va="center",
                     color="grey")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    bigram_matrix, vocabulary = create_bigram_matrix(names)
    plot_bigram_heatmap(bigram_matrix, vocabulary)


if __name__ == '__main__':
    main()
