def generate_bigrams(name):
    bigrams = []

    for i in range(len(name) - 1):
        bigrams.append((name[i], name[i + 1]))

    return bigrams


def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')

    for prev, next in generate_bigrams(names[0]):
        print(f"{prev} {next}")


if __name__ == '__main__':
    main()
