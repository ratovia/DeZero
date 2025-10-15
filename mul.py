from __future__ import annotations

from typing import Any

import numpy as np

from function import Function
from variable import Variable


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, 'inputs') or len(self.inputs) != 2:
            raise ValueError('Mul function requires exactly two inputs.')

        x0, x1 = self.inputs
        if x0.data is None or x1.data is None:
            raise ValueError('Input data must not be None.')

        gx0 = gy * x1.data
        gx1 = gy * x0.data
        return gx0, gx1


def mul(x0: Any, x1: Any) -> Variable:
    x0_var = Variable._ensure_variable(x0)
    x1_var = Variable._ensure_variable(x1)
    return Mul()(x0_var, x1_var)


Variable.__mul__ = mul  # type: ignore[assignment]
