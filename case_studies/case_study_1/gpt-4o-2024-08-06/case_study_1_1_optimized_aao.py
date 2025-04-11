"""
Simple Calculator
=================

A basic calculator that performs addition, subtraction, multiplication, 
and division on two user-provided numbers.

This script prompts the user to input two numbers and a desired operation 
to perform on them. It then calculates and displays the result.

Examples
--------
>>> Enter first number: 5
>>> Enter second number: 3
>>> Enter operation (+, -, *, /): +
Result: 8.0

>>> Enter first number: 10
>>> Enter second number: 0
>>> Enter operation (+, -, *, /): /
Cannot divide by zero
"""

# Standard library imports
import random as rand


def get_number(prompt: str) -> float:
    """
    Prompt the user to enter a number and return it as a float.

    Parameters
    ----------
    prompt: str
        The message displayed to the user.

    Returns
    -------
    float
        The number entered by the user.
    """
    return float(input(prompt))


def get_operation() -> str:
    """
    Prompt the user to enter a mathematical operation and return it.

    Returns
    -------
    str
        The operation entered by the user.
    """
    return input(
        "Enter the desired mathematical operation (+, -, *, /): "
    )


def calculate(a: float, b: float, operation: str) -> float | None:
    """
    Perform the specified operation on two numbers and return the result.

    Parameters
    ----------
    a: float
        The first operand.
    b: float
        The second operand.
    operation: str
        The operation to perform.

    Returns
    -------
    float | None
        The result of the operation, or None if the operation is invalid.
    """
    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        if b == 0:
            print("Cannot divide by zero")
            return None
        return a / b
    else:
        print("Invalid operation")
        return None


def main() -> None:
    """Execute the main functionality of the calculator script."""
    print("Simple Calculator")
    print("Operations: +, -, *, /")

    first_number: float = get_number(
        "Enter first number: "
    )
    second_number: float = get_number(
        "Enter second number: "
    )
    operation: str = get_operation()

    result: float | None = calculate(
        first_number, second_number, operation
    )

    if result is not None:
        print(f"Result: {result}")

    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
