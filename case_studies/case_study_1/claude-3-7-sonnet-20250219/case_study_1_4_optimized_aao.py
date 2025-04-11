"""
Simple Calculator
================

A basic calculator that performs addition, subtraction, multiplication, and division.

This module provides a simple command-line calculator that takes two numbers and
an operation as input, then displays the result.

Examples
--------
>>> python calculator.py
Simple Calculator
Operations: +,-,*,/
Enter first number: 5
Enter second number: 3
Enter operation: +
Result: 8.0
Thank you for using the calculator!
"""


import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def get_first_number() -> float:
    """
    Get the first number from user input.

    Returns
    -------
    float
        The first number entered by the user.
    """
    return float(input("Enter first number: "))


def get_second_number() -> float:
    """
    Get the second number from user input.

    Returns
    -------
    float
        The second number entered by the user.
    """
    return float(input("Enter second number: "))


def get_operation() -> str:
    """
    Get the desired operation from user input.

    Returns
    -------
    str
        The operation symbol entered by the user.
    """
    return input("Enter operation: ")


def calculate(a: float, b: float, operation: str) -> float | None:
    """
    Perform the calculation based on the given numbers and operation.

    Parameters
    ----------
    a: float
        The first number.
    b: float
        The second number.
    operation: str
        The operation to perform (+, -, *, /).

    Returns
    -------
    float | None
        The result of the calculation or None if the operation is invalid.
    """
    logging.info(f"Calculating {a} {operation} {b}")

    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        if b == 0:
            logging.error("Cannot divide by zero")
            print("Cannot divide by zero")
            return None
        return a / b
    else:
        logging.error(f"Invalid operation: {operation}")
        print("Invalid operation")
        return None


def main() -> None:
    """Executes the main functionality of the calculator."""
    print("Simple Calculator")
    print("Operations: +,-,*,/")

    a = get_first_number()
    b = get_second_number()
    operation = get_operation()

    result = calculate(a, b, operation)

    if result is not None:
        print(f"Result: {result}")

    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
