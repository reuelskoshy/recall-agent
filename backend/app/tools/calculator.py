"""calculator: exact arithmetic, because LLMs are unreliable at it.

Expressions are evaluated by walking Python's AST with an allow-list of
operators and functions. `eval()` is never used: it would let the model (or a
prompt injection) run arbitrary code.
"""

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolError, ToolResult
from app.tools.registry import register_tool

Number = int | float

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 1000
MAX_FACTORIAL = 170  # 171! overflows a float
MAX_INT_DIGITS = 400

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _factorial(n: Number) -> int:
    if not float(n).is_integer() or n < 0:
        raise ToolError("factorial() needs a non-negative whole number.")
    if n > MAX_FACTORIAL:
        raise ToolError(f"factorial() is limited to n <= {MAX_FACTORIAL}.")
    return math.factorial(int(n))


_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,  # log(x) natural, log(x, base)
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": _factorial,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def evaluate(expression: str) -> Number:
    """Safely evaluate an arithmetic expression. Raises ToolError on anything unsupported."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError(f"Expression is longer than {MAX_EXPRESSION_CHARS} characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ToolError(f"Could not parse expression: {expression!r}") from None
    try:
        return _eval(tree.body)
    except ZeroDivisionError:
        raise ToolError("Division by zero.") from None
    except (ValueError, OverflowError) as exc:
        raise ToolError(f"Math error: {exc}.") from None


def _eval(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool):
            raise ToolError("Booleans are not numbers here.")
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))  # type: ignore[no-any-return]
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitXor):
            raise ToolError("Use ** for powers (^ is not exponentiation).")
        if type(node.op) in _BINARY_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError(f"Exponent is limited to {MAX_EXPONENT}.")
            result = _BINARY_OPS[type(node.op)](left, right)
            if isinstance(result, complex):
                raise ToolError("Result is a complex number, which is not supported.")
            if isinstance(result, int) and abs(result).bit_length() > MAX_INT_DIGITS * 3.33:
                raise ToolError("Result is too large.")
            return result  # type: ignore[no-any-return]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))

    allowed = ", ".join(sorted(_FUNCTIONS) + sorted(_CONSTANTS))
    raise ToolError(
        f"Unsupported syntax: {ast.unparse(node)!r}. "
        f"Use numbers, + - * / // % **, parentheses, and: {allowed}."
    )


def format_number(value: Number) -> str:
    """Readable output: integers exactly, floats to 12 significant digits."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


class CalculatorInput(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(
        description="Arithmetic expression, e.g. '1234 * 5678', '(3 + 4) ** 2', "
        "'sqrt(2) * pi', 'log(100, 10)'. Use ** for powers.",
        max_length=MAX_EXPRESSION_CHARS,
    )


@register_tool
class CalculatorTool(Tool[CalculatorInput]):
    """Exact arithmetic."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression exactly. Use this for any calculation "
        "instead of doing math in your head."
    )
    input_model = CalculatorInput

    async def run(self, args: CalculatorInput, ctx: ToolContext) -> ToolResult:
        """Evaluate the expression."""
        value = evaluate(args.expression)
        result = format_number(value)
        return ToolResult(
            content=f"{args.expression} = {result}",
            data={"expression": args.expression, "result": result},
        )
