import numpy as np
import matplotlib.pyplot as plt
import graphviz
from typing import Set, Tuple, List, TypeAlias
import math


class Value:

    def __init__(self, data: float, label: str = '') -> None:
        self.data = data
        self.label = label
        self._prev = set()
        self._op: chr = ''
        self.gradient = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other) -> "Value":
        if isinstance(other, float):
            other = Value(float(other), str(other))

        new_label = "logit" if "b" in {self.label, other.label
                                       } else self.label + " + " + other.label
        result = Value(self.data + other.data, new_label)
        result._prev = {self, other}
        result._op = '+'

        def add_backward():
            self.gradient += result.gradient
            other.gradient += result.gradient

        result._backward = add_backward

        return result

    def __radd__(self, other) -> "Value":
        return self.__add__(other)

    def __mul__(self, other) -> "Value":
        if isinstance(other, float):
            other = Value(float(other), str(other))

        result = Value(self.data * other.data, self.label + other.label)
        result._prev = {self, other}
        result._op = '*'

        def mul_backward():
            self.gradient += other.data * result.gradient
            other.gradient += self.data * result.gradient

        result._backward = mul_backward

        return result

    def __rmul__(self, other) -> "Value":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Value":
        if isinstance(other, float):
            other = Value(float(other), str(other))

        result = Value(self.data / other.data,
                       f"({self.label} / {other.label})")
        result._prev = {self, other}
        result._op = "/"

        return result

    def __pow__(self, exponent: float) -> "Value":
        result = Value(self.data**exponent, f"({self.label} ** {exponent})")
        result._prev = {self}
        result._op = f"**{exponent}"

        return result

    def exp(self) -> "Value":
        result = Value(np.exp(self.data), f"exp({self.label})")
        result._prev = {self}
        result._op = "exp"

        return result

    def tanh(self) -> "Value":
        result = Value(math.tanh(self.data), "L")
        result._prev = {self}
        result._op = "tanh"

        def tanh_backward():
            self.gradient += (1 - result.data**2) * result.gradient

        result._backward = tanh_backward

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
    x = Value(2.0, label='x')

    expected = Value(4.0)

    actuals = {
        'actual_sum_l': x + 2.0,
        'actual_sum_r': 2.0 + x,
        'actual_mul_l': x * 2.0,
        'actual_mul_r': 2.0 * x,
        'actual_div_r': (x + 6.0) / 2.0,
        'actual_pow_l': x**2,
        'actual_exp_e': x**2,
    }

    assert x.exp().data == np.exp(
        2
    ), f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, f'Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}.'

    print('All tests passed!')


if __name__ == '__main__':
    main()
