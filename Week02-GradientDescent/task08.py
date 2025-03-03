import numpy as np
import matplotlib.pyplot as plt
import graphviz
from typing import Set, Tuple, List, TypeAlias


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


def topological_sort(nodes: Set[Node], edges: Set[Edge]) -> List[Node]:
    graph = {node: [] for node in nodes}
    for parent, child in edges:
        graph[parent].append(child)

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
    sorted_nodes = topological_sort(nodes, edges)
    for idx, n in enumerate(sorted_nodes):
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        letter_label = chr(ord('a') +
                           idx) if idx < len(sorted_nodes) - 1 else 'L'
        data_label = f"data: {n.data:.4f}"
        dot.node(name=uid,
                 label=f"{{ {letter_label} | {data_label} }}",
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
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    t = Value(-2)
    result = (x * y + z) * t

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
