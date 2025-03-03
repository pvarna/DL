import numpy as np
import matplotlib.pyplot as plt
import graphviz
from typing import Set, Tuple, List, TypeAlias


class Value:
    data: float = 0.0
    _prev: Set["Value"] = set()
    _op: chr = ''
    gradient: float = 0.0
    label: str = ""

    def __init__(self, data: float, label: str) -> None:
        self.data = data
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        result = Value(self.data + other.data,
                       self.label + " + " + other.label)
        result._prev = {self, other}
        result._op = '+'
        return result

    def __mul__(self, other: "Value") -> "Value":
        result = Value(self.data * other.data, self.label + other.label)
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


def manual_der(root: Node) -> None:
    nodes, edges = trace(root)
    ordered_nodes = topological_sort(nodes, edges)

    for node in ordered_nodes:
        node.gradient = 0.0

    root.gradient = 1.0

    for node in reversed(ordered_nodes):
        child_grad = node.gradient
        if node._op == '+':
            # + => copy gradient to parents:
            # parent1.grad = current.grad
            # parent2.grad = current.grad
            p1, p2 = list(node._prev)
            p1.gradient = child_grad
            p2.gradient = child_grad

        elif node._op == '*':
            # * => multiply value of other parent with current gradient:
            # parent1.grad = parent2.value * current.grad
            # parent2.grad = parent1.value * current.grad
            p1, p2 = list(node._prev)
            p1.gradient = p2.data * child_grad
            p2.gradient = p1.data * child_grad


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result',
                           format='svg',
                           graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    sorted_nodes = topological_sort(nodes, edges)
    for idx, n in enumerate(sorted_nodes):
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        letter_label = n.label if idx < len(sorted_nodes) - 1 else "logit"
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
    x1 = Value(2.0, "x1")
    w1 = Value(-3.0, "w1")

    x2 = Value(0.0, "x2")
    w2 = Value(1.0, "w2")

    b = Value(6.7, "b")

    logit = x1 * w1 + x2 * w2 + b

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(logit).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
