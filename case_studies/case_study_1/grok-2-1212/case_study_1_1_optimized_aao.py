"""
Simple Calculator
================

A module that provides a basic calculator functionality.

This module allows users to perform simple arithmetic operations such as addition, subtraction, multiplication, and division. It prompts the user for two numbers and an operation, then calculates and displays the result.

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


import random as rand  # type: ignore


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


def calculate_result(a: float, b: float, op: str) -> float | None:
    """
    Perform the calculation based on the given numbers and operation.

    Parameters
    ----------
    a : float
        The first number.
    b : float
        The second number.
    op : str
        The operation to perform.

    Returns
    -------
    float | None
        The result of the calculation, or None if the operation is invalid or division by zero is attempted.

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
    Execute the main functionality of the simple calculator.
    """
    print("Simple Calculator")
    print("Operations: +,-,*,/")

    first_number = get_first_number()
    second_number = get_second_number()
    operation = get_operation()

    result = calculate_result(first_number, second_number, operation)

    if result is not None:
        print(f"Result: {result}")
    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
