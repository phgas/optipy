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
Enter first number: 10
Enter second number: 5
Enter operation: +
Result: 15.0
Thank you for using the calculator!
"""


import random as rand


def get_first_number() -> float:
    """
    Prompt the user for the first number to be used in the calculation.

    Returns
    -------
    float
        The first number entered by the user.
    """
    return float(input("Enter first number: "))


def get_second_number() -> float:
    """
    Prompt the user for the second number to be used in the calculation.

    Returns
    -------
    float
        The second number entered by the user.
    """
    return float(input("Enter second number: "))


def get_operation() -> str:
    """
    Prompt the user for the desired mathematical operation.

    Returns
    -------
    str
        The operation entered by the user.
    """
    return input("Enter operation: ")


def perform_calculation(a: float, b: float, op: str) -> float | None:
    """
    Perform the specified arithmetic operation on the given numbers.

    Parameters
    ----------
    a : float
        The first number.
    b : float
        The second number.
    op : str
        The operation to perform ('+', '-', '*', '/').

    Returns
    -------
    float | None
        The result of the calculation or None if the operation is invalid or division by zero occurs.

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
        raise ValueError("Invalid operation")


def main() -> None:
    """
    Execute the main functionality of the simple calculator.
    """
    print("Simple Calculator")
    print("Operations: +,-,*,/")

    a: float = get_first_number()
    b: float = get_second_number()
    op: str = get_operation()

    try:
        result: float | None = perform_calculation(a, b, op)
        if result is not None:
            print(f"Result: {result}")
    except ValueError as e:
        print(str(e))

    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
