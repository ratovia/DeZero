from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from function import Function


class Variable:
    def __init__(self, data: Optional[np.ndarray]) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data: Optional[np.ndarray] = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func: Function) -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        if self.creator is not None:
            funcs.append(self.creator)

        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            if y.grad is None:
                raise ValueError('Gradient of output is None.')
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
