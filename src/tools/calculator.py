import ast
import operator
from typing import Any


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        operand = _eval_node(node.operand)
        return _OPERATORS[type(node.op)](operand)

    raise ValueError("只支持数字和基本数学运算：+ - * / // % ** ()")


def calculator(input: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        tree = ast.parse(input, mode="eval")
        result: Any = _eval_node(tree)
    except Exception as exc:
        return f"计算失败：{exc}"

    return f"{input} = {result}"
