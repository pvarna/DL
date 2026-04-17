import math
import numpy as np
from typing import Set, Tuple, List, TypeAlias


class Value:

    def __init__(self, data: float, label: str = '') -> None:
        self.data = data
        self.label = label
        self._prev = set()
        self._op: str = ''
        self.gradient = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other) -> "Value":
        if isinstance(other, (float, int)):
            other = Value(float(other))
        result = Value(self.data + other.data)
        result._prev = {self, other}
        result._op = '+'

        def add_backward():
            self.gradient += result.gradient
            other.gradient += result.gradient

        result._backward = add_backward
        return result

    def __radd__(self, other) -> "Value":
        return self.__add__(other)

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __mul__(self, other) -> "Value":
        if isinstance(other, (float, int)):
            other = Value(float(other))
        result = Value(self.data * other.data)
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
        if isinstance(other, (float, int)):
            other = Value(float(other))
        result = Value(self.data / other.data)
        result._prev = {self, other}
        result._op = '/'

        def div_backward():
            self.gradient += (1 / other.data) * result.gradient
            other.gradient += (-self.data / other.data**2) * result.gradient

        result._backward = div_backward
        return result

    def __pow__(self, exponent: float) -> "Value":
        result = Value(self.data**exponent)
        result._prev = {self}
        result._op = f'**{exponent}'

        def pow_backward():
            self.gradient += (exponent *
                              self.data**(exponent - 1)) * result.gradient

        result._backward = pow_backward
        return result

    def exp(self) -> "Value":
        result = Value(np.exp(self.data))
        result._prev = {self}
        result._op = 'e'

        def exp_backward():
            self.gradient += result.data * result.gradient

        result._backward = exp_backward
        return result

    def tanh(self) -> "Value":
        result = Value(math.tanh(self.data))
        result._prev = {self}
        result._op = 'tanh'

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


def top_sort(nodes: Set[Node]) -> List[Node]:
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


class Neuron:

    def __init__(self, number_of_inputs: int) -> None:
        self.w = [
            Value(np.random.uniform(-1, 1)) for _ in range(number_of_inputs)
        ]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x: list[float]) -> Value:
        weighted_sum = sum(
            (weight * input_value for weight, input_value in zip(self.w, x)),
            start=Value(0.0))
        logit = weighted_sum + self.b
        return logit.tanh()


def main() -> None:
    np.random.seed(42)
    n = Neuron(2)
    x = [2.0, 3.0]
    print(n(x))


if __name__ == '__main__':
    main()
