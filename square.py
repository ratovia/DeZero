from __future__ import annotations

import numpy as np

from function import Function
from variable import Variable


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        if x is None:
            raise ValueError('Input data must not be None.')
        return 2 * x * gy


def square(x: Variable) -> Variable:
    return Square()(x)
