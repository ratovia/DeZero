from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from function import Function
from variable import Variable


class Add(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        if not xs:
            raise ValueError('Add function requires at least one input.')

        total = xs[0]
        for x in xs[1:]:
            total = total + x
        return total

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...]:
        if not hasattr(self, 'inputs'):
            raise ValueError('Inputs are not set for this function.')
        return tuple(gy for _ in range(len(self.inputs)))


def add(*inputs: Any) -> Variable:
    if len(inputs) == 0:
        raise ValueError('Add function requires at least one input.')

    if len(inputs) == 1 and isinstance(inputs[0], Sequence):
        sequence_inputs = inputs[0]
        variables = [Variable._ensure_variable(value) for value in sequence_inputs]
    else:
        variables = [Variable._ensure_variable(value) for value in inputs]

    if not variables:
        raise ValueError('Add function requires at least one input.')

    return Add()(*variables)  # type: ignore[arg-type]


Variable.__add__ = add  # type: ignore[assignment]
