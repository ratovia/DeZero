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
        self.generation: int = 0

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []

        def add_func(f: Function) -> None:
            funcs.append(f)
            funcs.sort(key=lambda func: func.generation)

        if self.creator is not None:
            add_func(self.creator)

        while funcs:
            f = funcs.pop()
            output_refs = getattr(f, 'outputs', None)
            if output_refs is None:
                raise ValueError('Function outputs are missing.')

            outputs = [output_ref() for output_ref in output_refs]
            if any(output is None for output in outputs):
                raise ValueError('Function output has been deallocated.')

            gys = [output.grad for output in outputs]
            if any(gy is None for gy in gys):
                raise ValueError('Gradient of output is None.')

            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            inputs = getattr(f, 'inputs', None)
            if inputs is None:
                raise ValueError('Function inputs are missing.')

            for x, gx in zip(inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

    def clear_grad(self) -> None:
        """Reset the stored gradient."""

        self.grad = None
