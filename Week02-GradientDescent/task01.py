import numpy as np
import matplotlib.pyplot as plt


class Value:
    data: float = 0.0

    def __init__(self, data: float) -> None:
        self.data = data

    def __repr__(self) -> str:
        return f"Value(data={self.data})"


def main() -> None:
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)


if __name__ == '__main__':
    main()
