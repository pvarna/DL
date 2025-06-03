import torch
import math

SEED = 42


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


def normalize_row_wise(bigram_matrix):
    for i in range(bigram_matrix.size(0)):
        row_sum = bigram_matrix[i].sum()

        for j in range(len(bigram_matrix[i])):
            bigram_matrix[i, j] = bigram_matrix[i, j] / row_sum


def create_bigram_matrix(names):
    vocabulary = get_vocabulary(names)
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}

    N = len(vocabulary)
    bigram_matrix = torch.zeros((N, N), dtype=torch.float)

    for name in names:
        bigrams = generate_bigrams(name)
        for prev, next in bigrams:
            i = char_to_idx[prev]
            j = char_to_idx[next]
            bigram_matrix[i, j] += 1

    return bigram_matrix, vocabulary


def sample_names(bigram_matrix, vocabulary, num_names):
    torch.manual_seed(SEED)

    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}

    words = []
    for _ in range(num_names):
        word = ""
        idx = char_to_idx["."]

        while True:
            probs = bigram_matrix[idx]
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_idx]

            if next_char == ".":
                break

            word += next_char
            idx = next_idx

        words.append(word)

    return words


def compute_nll(bigram_matrix, names, vocabulary):
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    total_log_likelihood = 0.0
    count = 0

    for name in names:
        bigrams = generate_bigrams(name)
        for prev, next in bigrams:
            i = char_to_idx[prev]
            j = char_to_idx[next]
            prob = bigram_matrix[i, j].item()

            prob = max(prob, 0.00001)  # avoid log(0)
            total_log_likelihood += -math.log(prob)
            count += 1

    average_nll = total_log_likelihood / count
    return average_nll


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    bigram_matrix, vocabulary = create_bigram_matrix(names)
    normalize_row_wise(bigram_matrix)

    loss = compute_nll(bigram_matrix, names, vocabulary)
    print(f"Loss: {loss}")

    # Yes, a naive model would assign equal probability to all characters
    # Vocabulary size = 27, p(char) = 1/27, NLL = -ln(1/27) ~ 3.3
    # Out loss is ~2.45, which is lower


if __name__ == '__main__':
    main()
