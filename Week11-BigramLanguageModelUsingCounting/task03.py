def generate_bigrams(name):
    bigrams = []

    bigrams.append(("<S>", name[0]))

    for i in range(len(name) - 1):
        bigrams.append((name[i], name[i + 1]))

    bigrams.append((name[-1], "<E>"))

    return bigrams


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    for i in range(3):
        for prev, next in generate_bigrams(names[i]):
            print(f"{prev} {next}")


if __name__ == '__main__':
    main()
