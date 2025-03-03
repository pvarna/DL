import numpy as np
import matplotlib.pyplot as plt
from typing import Set, Tuple


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


def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    nodes: Set[Value] = set()
    edges: Set[Tuple[Value, Value]] = set()

    def helper(node: Value) -> None:
        if node not in nodes:
            nodes.add(node)
            for parent in node._prev:
                edges.add((parent, node))
                helper(parent)

    helper(root)
    return nodes, edges


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')


if __name__ == '__main__':
    main()
