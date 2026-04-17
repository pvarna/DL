import numpy as np
import matplotlib.pyplot as plt
import graphviz
from typing import Set, Tuple


class Value:

    def __init__(self, data: float) -> None:
        self.data = data
        self._prev = set()
        self._op: chr = ''

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


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result',
                           format='svg',
                           graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ data: {n.data} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
