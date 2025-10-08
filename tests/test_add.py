import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from add import add
from variable import Variable


class AddTest(unittest.TestCase):
    def test_forward_with_multiple_inputs(self):
        x1 = Variable(np.array(1.0))
        x2 = Variable(np.array(2.0))
        x3 = Variable(np.array(3.0))
        y = add(x1, x2, x3)
        expected = np.array(6.0)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward_accumulates_gradient_for_duplicate_inputs(self):
        x = Variable(np.array(1.0))
        y = add(x, x, x)
        y.backward()
        expected = np.array(3.0)
        self.assertTrue(np.allclose(x.grad, expected))

    def test_forward_with_sequence_input(self):
        x1 = Variable(np.array([1.0, 2.0]))
        x2 = Variable(np.array([3.0, 4.0]))
        y = add([x1, x2])
        expected = np.array([4.0, 6.0])
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward_after_clearing_grad(self):
        x = Variable(np.array(1.0))

        y1 = add(x, x)
        y1.backward()
        self.assertTrue(np.allclose(x.grad, np.array(2.0)))

        x.clear_grad()

        y2 = add(x, add(x, x))
        y2.backward()
        self.assertTrue(np.allclose(y2.data, np.array(3.0)))
        self.assertTrue(np.allclose(x.grad, np.array(3.0)))


if __name__ == '__main__':
    unittest.main()
