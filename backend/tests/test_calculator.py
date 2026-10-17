"""Unit tests for the calculator's safe evaluator."""

import pytest

from app.tools.base import ToolError
from app.tools.calculator import evaluate, format_number


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1234 * 5678", "7006652"),
        ("(3 + 4) ** 2", "49"),
        ("10 / 4", "2.5"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("-5 + +2", "-3"),
        ("sqrt(16)", "4"),
        ("log(100, 10)", "2"),
        ("round(pi, 4)", "3.1416"),
        ("factorial(10)", "3628800"),
        ("0.1 + 0.2", "0.3"),
        ("2 ** 100", "1267650600228229401496703205376"),
    ],
)
def test_evaluates_supported_expressions(expression: str, expected: str) -> None:
    assert format_number(evaluate(expression)) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os').system('rm -rf /')", "Unsupported syntax"),
        ("().__class__.__bases__", "Unsupported syntax"),
        ("open('/etc/passwd')", "Unsupported syntax"),
        ("x + 1", "Unsupported syntax"),
        ("2 ^ 3", r"Use \*\*"),
        ("1 / 0", "Division by zero"),
        ("2 ** 100000", "Exponent is limited"),
        ("(2 ** 999) ** 999", "too large"),
        ("factorial(1000)", "limited"),
        ("sqrt(-1)", "Math error"),
        ("(-8) ** 0.5", "complex"),
        ("True + 1", "Booleans"),
        ("1 +", "Could not parse"),
        ("1" * 201, "longer than"),
    ],
)
def test_rejects_unsafe_or_invalid_expressions(expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        evaluate(expression)
