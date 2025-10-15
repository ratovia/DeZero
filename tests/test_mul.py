import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mul import mul
from variable import Variable


class MulTest(unittest.TestCase):
    def test_forward(self):
        x1 = Variable(np.array(2.0))
        x2 = Variable(np.array(3.0))
        y = mul(x1, x2)
        self.assertTrue(np.allclose(y.data, np.array(6.0)))

    def test_backward(self):
        x1 = Variable(np.array(2.0))
        x2 = Variable(np.array(3.0))
        y = mul(x1, x2)
        y.backward()
        self.assertTrue(np.allclose(x1.grad, np.array(3.0)))
        self.assertTrue(np.allclose(x2.grad, np.array(2.0)))

    def test_operator_overload_with_scalar(self):
        x = Variable(np.array(4.0))
        y = x * 5
        y.backward()
        self.assertTrue(np.allclose(y.data, np.array(20.0)))
        self.assertTrue(np.allclose(x.grad, np.array(5.0)))


if __name__ == '__main__':
    unittest.main()
