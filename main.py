import argparse
import sys
from typing import Any, Dict

import numpy as np

from add import Add, add
from exp import exp
from square import Square, square
from variable import Variable


def _sanitize_expression(expr: str) -> str:
    return expr.replace("^", "**")


def _to_variable(value: Any) -> Variable:
    if isinstance(value, Variable):
        return value
    try:
        return Variable(np.array(value))
    except Exception as exc:
        raise TypeError("Variableへ変換できませんでした。") from exc


def evaluate_expression(expr: str, env: Dict[str, Any]) -> tuple[Variable, str]:
    sanitized = _sanitize_expression(expr)
    try:
        result = eval(sanitized, {"__builtins__": {}}, env)
    except Exception as exc:
        raise ValueError(f"式の評価に失敗しました: {expr}") from exc

    try:
        return _to_variable(result), sanitized
    except TypeError as exc:
        raise ValueError("式はVariableまたは数値を返す必要があります。") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="DeZero simple forward/backward demo")
    parser.add_argument(
        "expression",
        nargs="?",
        default="x",
        help="xを含む式 (例: 1 + x^2, square(x))",
    )
    parser.add_argument(
        "--x-value",
        type=float,
        default=0.5,
        help="xに設定する値 (デフォルト: 0.5)",
    )
    args = parser.parse_args()

    x = Variable(np.array(args.x_value))
    env: Dict[str, Any] = {
        "x": x,
        "Variable": Variable,
        "square": square,
        "Square": Square,
        "add": add,
        "Add": Add,
        "exp": exp,
        "np": np,
    }

    try:
        y, sanitized_expression = evaluate_expression(args.expression, env)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    y.backward()
    print(f"入力式: y={args.expression}")
    print(f"x={x.data}の時、y={y.data}、傾き={x.grad}")


if __name__ == "__main__":
    main()
