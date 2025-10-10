from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from variable import Variable


def as_array(x: Any) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        if len(inputs) == 1 and isinstance(inputs[0], Sequence):
            sequence_inputs = inputs[0]
            if all(isinstance(x, Variable) for x in sequence_inputs):
                inputs = tuple(sequence_inputs)  # type: ignore[assignment]

        xs = [x.data for x in inputs]
        for x in xs:
            if x is None:
                raise ValueError('Input data must not be None.')

        self.generation = max((x.generation for x in inputs), default=0)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs = list(inputs)
        self.outputs = outputs
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *xs: np.ndarray) -> Tuple[np.ndarray, ...] | np.ndarray:
        raise NotImplementedError()

    def backward(self, *gys: np.ndarray) -> Tuple[np.ndarray, ...] | np.ndarray:
        raise NotImplementedError()
