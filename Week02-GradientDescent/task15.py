import numpy as np
import matplotlib.pyplot as plt
import graphviz
from typing import Set, Tuple, List, TypeAlias
import math


class Value:
    data: float = 0.0
    _prev: Set["Value"] = set()
    _op: str = ''
    gradient: float = 0.0
    label: str = ""

    def __init__(self, data: float, label: str) -> None:
        self.data = data
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        result = Value(self.data + other.data,
                       self.label + " + " + other.label)
        result._prev = {self, other}
        result._op = '+'

        def add_backward():
            self.gradient = result.gradient
            other.gradient = result.gradient

        result._backward = add_backward

        return result

    def __mul__(self, other: "Value") -> "Value":
        result = Value(self.data * other.data, self.label + other.label)
        result._prev = {self, other}
        result._op = '*'
        return result

    def tanh(self) -> "Value":
        result = Value(math.tanh(self.data), "L")
        result._prev = {self}
        result._op = "tanh"
        return result

    def backward(self) -> None:
        nodes, _ = trace(self)
        sorted_nodes = top_sort(nodes)

        self.gradient = 1.0

        for node in reversed(sorted_nodes):
            node._backward()


Node: TypeAlias = Value
Edge: TypeAlias = Tuple[Value, Value]


def trace(root: Value) -> Tuple[Set[Node], Set[Edge]]:
    nodes: Set[Node] = set()
    edges: Set[Edge] = set()

    def helper(node: Node) -> None:
        if node not in nodes:
            nodes.add(node)
            for parent in node._prev:
                edges.add((parent, node))
                helper(parent)

    helper(root)
    return nodes, edges


def top_sort(nodes: List[Node]) -> List[Node]:
    graph = {node: [] for node in nodes}
    for node in nodes:
        for parent in node._prev:
            graph[parent].append(node)

    L = []
    permanent_mark = set()
    temporary_mark = set()

    def visit(node: Node) -> None:
        if node in permanent_mark:
            return
        if node in temporary_mark:
            raise ValueError("Cycle detected")

        temporary_mark.add(node)

        for child in graph[node]:
            visit(child)

        temporary_mark.remove(node)
        permanent_mark.add(node)

        L.insert(0, node)

    for node in nodes:
        if node not in permanent_mark:
            visit(node)

    return L


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result',
                           format='svg',
                           graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        letter_label = n.label
        data_label = f"data: {n.data:.4f}"
        gradient_label = f"grad: {n.gradient:.4f}"
        dot.node(
            name=uid,
            label=f"{{ {letter_label} | {data_label} | {gradient_label} }}",
            shape='record')
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
    x = Value(5.0, "x")
    y = Value(10.0, "y")

    z = x + y
    z.label = "z"

    z.backward()

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(z).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
