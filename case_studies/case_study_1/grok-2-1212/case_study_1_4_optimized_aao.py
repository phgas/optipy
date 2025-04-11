"""
Simple Calculator
================

A basic calculator module that performs arithmetic operations.

This module provides functionality to perform addition, subtraction, 
multiplication, and division on two numbers provided by the user.

Examples
--------
>>> from simple_calculator import main
>>> main()
Simple Calculator
Operations: +,-,*,/
Enter first number: 5
Enter second number: 3
Enter operation: +
Result: 8.0
Thank you for using the calculator!
"""

import random as rand


def get_first_number() -> float:
    """
    Prompt the user for the first number.

    Returns
    -------
    float
        The first number entered by the user.
    """
    return float(input("Enter first number: "))


def get_second_number() -> float:
    """
    Prompt the user for the second number.

    Returns
    -------
    float
        The second number entered by the user.
    """
    return float(input("Enter second number: "))


def get_operation() -> str:
    """
    Prompt the user for the desired operation.

    Returns
    -------
    str
        The operation entered by the user.
    """
    return input("Enter operation (+, -, *, /): ")


def perform_calculation(a: float, b: float, op: str) -> float | None:
    """
    Perform the specified arithmetic operation on two numbers.

    Parameters
    ----------
    a : float
        The first operand.
    b : float
        The second operand.
    op : str
        The operation to perform.

    Returns
    -------
    float | None
        The result of the operation or None if the operation is invalid or division by zero.

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
    Execute the main functionality of the calculator.
    """
    print("Simple Calculator")
    print("Operations: +,-,*,/")

    first_number: float = get_first_number()
    second_number: float = get_second_number()
    operation: str = get_operation()

    result: float | None = perform_calculation(
        first_number, second_number, operation)

    if result is not None:
        print(f"Result: {result}")
    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
