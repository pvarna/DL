def main():
    names_file = open("../DATA/names.txt")
    names = names_file.read().split('\n')
    shortest_word = min(names, key=len)
    longest_word = max(names, key=len)

    print(names[:10])
    print(f"Total number of words: {len(names)}")
    print(f"Length of shortest word: {len(shortest_word)}")
    print(f"Length of lonsest word: {len(longest_word)}")


if __name__ == '__main__':
    main()
