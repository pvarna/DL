def generate_bigrams(name):
    bigrams = []

    bigrams.append(("<S>", name[0]))

    for i in range(len(name) - 1):
        bigrams.append((name[i], name[i + 1]))

    bigrams.append((name[-1], "<E>"))

    return bigrams


def count_bigrams(names):
    bigram_counts = dict()
    for name in names:
        bigrams = generate_bigrams(name)
        for bigram in bigrams:
            if bigram in bigram_counts:
                bigram_counts[bigram] += 1
            else:
                bigram_counts[bigram] = 1

    bigram_counts = sorted(bigram_counts.items(),
                           key=lambda x: x[1],
                           reverse=True)

    return bigram_counts


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    bigram_counts = count_bigrams(names)
    print(bigram_counts[:15])


if __name__ == '__main__':
    main()
