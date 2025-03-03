import numpy as np
import matplotlib.pyplot as plt
from typing import Set


class Value:
    data: float = 0.0
    _prev: Set["Value"] = set()
    _op: chr = ''

    def __init__(self, data: float) -> None:
        self.data = data

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        result = Value(self.data + other.data)
        result._prev = {self, other}
        result._op = '+'
        return result

    def __mul__(self, other: "Value") -> "Value":
        result = Value(self.data * other.data)
        result._prev = {self, other}
        result._op = '*'
        return result


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)


if __name__ == '__main__':
    main()
