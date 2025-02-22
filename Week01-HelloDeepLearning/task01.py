import numpy as np


def main():
    baseball = [180, 215, 210, 210, 188, 176, 209, 200]
    np_baseball = np.array(baseball)
    print(f"Baseball array: {np_baseball}")
    print(f"Type of baseball array: {type(np_baseball)}")


if __name__ == '__main__':
    main()
