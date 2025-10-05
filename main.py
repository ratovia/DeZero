import numpy as np

from exp import exp
from square import square
from variable import Variable


def main() -> None:
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(f"y.data: {y.data}")
    print(f"x.grad: {x.grad}")


if __name__ == "__main__":
    main()
