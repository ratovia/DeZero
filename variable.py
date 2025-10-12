from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from function import Function


class Variable:
    def __init__(self, data: np.ndarray, name: Optional[str] = None) -> None:
        if data is None:
            raise TypeError('None is not supported')
        if not isinstance(data, np.ndarray):
            raise TypeError(f'{type(data)} is not supported')

        self.data: np.ndarray = data
        self.name = name
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool = False) -> None:
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

            if not retain_grad:
                for output_ref in output_refs:
                    output = output_ref()
                    if output is not None:
                        output.grad = None

    def clear_grad(self) -> None:
        """Reset the stored gradient."""

        self.grad = None

    @staticmethod
    def _ensure_variable(value: Any) -> "Variable":
        if isinstance(value, Variable):
            return value
        return Variable(np.array(value))

    def __add__(self, other: Any) -> "Variable":
        from add import add as add_func

        other_var = self._ensure_variable(other)
        return add_func(self, other_var)

    def __radd__(self, other: Any) -> "Variable":
        return self.__add__(other)

    def __pow__(self, power: Any) -> "Variable":
        from square import square as square_func

        if power == 2:
            return square_func(self)
        raise ValueError("現在は指数2のみ対応しています。")

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
