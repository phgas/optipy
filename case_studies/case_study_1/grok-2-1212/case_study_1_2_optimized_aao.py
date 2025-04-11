"""
Simple Calculator
================

A basic calculator module that performs arithmetic operations.

This module provides functionality to perform addition, subtraction,
multiplication, and division on two numbers provided by the user.

Examples
--------
>>> from simple_calculator import calculate
>>> result = calculate(5, 3, '+')
>>> print(result)
8.0

>>> result = calculate(10, 2, '/')
>>> print(result)
5.0
"""


def get_number(prompt: str) -> float:
    """
    Prompts the user for a number and returns it as a float.

    Parameters
    ----------
    prompt: str
        The message to display to the user.

    Returns
    -------
    float
        The number entered by the user.
    """
    return float(input(prompt))


def get_operation() -> str:
    """
    Prompts the user for the desired mathematical operation.

    Returns
    -------
    str
        The operation entered by the user.
    """
    return input("Enter the desired mathematical operation (+, -, *, /): ")


def calculate(a: float, b: float, op: str) -> float | None:
    """
    Performs the specified arithmetic operation on two numbers.

    Parameters
    ----------
    a: float
        The first operand.
    b: float
        The second operand.
    op: str
        The operation to perform ('+', '-', '*', '/').

    Returns
    -------
    float | None
        The result of the operation, or None if the operation is invalid or division by zero is attempted.

    Raises
    ------
    ValueError
        If an invalid operation is provided.
    """
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            print("Cannot divide by zero")
            return None
        return a / b
    else:
        print("Invalid operation")
        return None


def main() -> None:
    """
    Executes the main functionality of the calculator.
    """
    print("Simple Calculator")
    print("Operations: +, -, *, /")

    first_number = get_number("Enter first number: ")
    second_number = get_number("Enter second number: ")
    operation = get_operation()

    result = calculate(first_number, second_number, operation)

    if result is not None:
        print(f"Result: {result}")
    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
