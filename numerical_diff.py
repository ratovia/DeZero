from __future__ import annotations

from typing import Callable

import numpy as np

from variable import Variable


def numerical_diff(
    f: Callable[[Variable], Variable],
    x: Variable,
    eps: float = 1e-4,
) -> np.ndarray:
    if x.data is None:
        raise ValueError('Input variable must contain data.')
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    if y0.data is None or y1.data is None:
        raise ValueError('Output variable must contain data.')
    return (y1.data - y0.data) / (2 * eps)
