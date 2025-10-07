from __future__ import annotations

import numpy as np

from function import Function
from variable import Variable


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'inputs') or not self.inputs:
            raise ValueError('Inputs are not set for this function.')
        x = self.inputs[0].data
        if x is None:
            raise ValueError('Input data must not be None.')
        return np.exp(x) * gy


def exp(x: Variable) -> Variable:
    return Exp()(x)
