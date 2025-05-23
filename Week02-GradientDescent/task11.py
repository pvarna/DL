import numpy as np
import matplotlib.pyplot as plt
import graphviz
from typing import Set, Tuple, List, TypeAlias


class Value:

    def __init__(self, data: float) -> None:
        self.data = data
        self._prev = set()
        self._op: chr = ''
        self.gradient = 0.0

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
        letter_label = chr(ord('a') +
                           idx) if idx < len(sorted_nodes) - 1 else 'L'
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


def update_values(nodes: Set[Value], learning_rate: float = 0.01) -> None:
    for node in nodes:
        # ???
        node.data += learning_rate * node.gradient  # doing gradient ascent in order to meet the values from the example
        # if we were doing gradient descent --> node.data -= learning_rate * node.gradient
        # maybe the confusion comes from the fact that the error is negative, which is not natural


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    t = Value(-2)
    result = (x * y + z) * t

    manual_der(result)

    print(f"Old L = {result.data}")

    nodes, _ = trace(result)
    update_values(nodes)

    result_new = (x * y + z) * t
    print(f"New L = {result_new.data}")

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    # draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
