from __future__ import annotations

import ast
import re


class UnsupportedExpression(ValueError):
    pass


class MalformedExpression(ValueError):
    pass


_ALLOWED_SYMBOLS = {"x", "y", "z"}
_LATEX_FRACTION = re.compile(
    r"\\(?:dfrac|tfrac|frac)\s*\{\s*([+-]?\d+(?:\.\d+)?)\s*\}\s*"
    r"\{\s*([+-]?\d+(?:\.\d+)?)\s*\}"
)


def expressions_equivalent(left: str, right: str) -> bool:
    """Compare a bounded arithmetic-expression subset without evaluating Python code."""
    try:
        left_expr = _parse_restricted(left)
        right_expr = _parse_restricted(right)
    except ImportError as exc:
        raise UnsupportedExpression("sympy_not_installed") from exc
    import sympy as sp

    try:
        return sp.cancel(left_expr - right_expr) == 0
    except (ArithmeticError, TypeError, ValueError, RecursionError) as exc:
        raise UnsupportedExpression("symbolic_comparison_limit") from exc


def _parse_restricted(value: str):
    import sympy as sp

    candidate = value.strip()
    if len(candidate) > 128:
        raise UnsupportedExpression("expression_too_long")
    if "=" in candidate:
        raise UnsupportedExpression("equations_not_supported")
    candidate = candidate.replace(r"\(", "").replace(r"\)", "")
    candidate = _LATEX_FRACTION.sub(r"((\1)/(\2))", candidate)
    if "\\" in candidate:
        raise UnsupportedExpression("latex_command_not_supported")
    candidate = candidate.replace("^", "**")
    try:
        tree = ast.parse(candidate, mode="eval")
    except SyntaxError as exc:
        raise MalformedExpression("malformed_expression") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 32:
        raise UnsupportedExpression("expression_too_complex")
    symbols = {name: sp.Symbol(name) for name in _ALLOWED_SYMBOLS}

    def build(node):
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in symbols:
                raise UnsupportedExpression("symbol_not_allowed")
            return symbols[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = build(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise MalformedExpression("division_by_zero")
                return left / right
            if not right.is_Integer or abs(int(right)) > 4:
                raise UnsupportedExpression("power_outside_supported_range")
            return left ** int(right)
        raise UnsupportedExpression("syntax_outside_restricted_arithmetic")

    return build(tree)
