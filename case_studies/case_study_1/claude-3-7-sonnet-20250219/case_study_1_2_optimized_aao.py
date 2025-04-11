"""
Simple Calculator
================

A basic calculator that performs addition, subtraction, multiplication, and division.

This module provides a simple command-line calculator that takes two numbers and
an operation as input, performs the calculation, and displays the result.

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


def get_numbers() -> tuple[float, float]:
    """
    Get two numbers from the user.

    Prompts the user to enter two numbers that will be used as operands
    in a mathematical operation.

    Returns
    -------
    tuple[float, float]
        A tuple containing the two numbers entered by the user.
    """
    first_number = float(input("Enter first number: "))
    second_number = float(input("Enter second number: "))
    return first_number, second_number


def get_operation() -> str:
    """
    Get the desired operation from the user.

    Prompts the user to enter the mathematical operation they want to perform.

    Returns
    -------
    str
        The operation symbol entered by the user (+, -, *, /).
    """
    return input("Enter operation: ")


def calculate(a: float, b: float, operation: str) -> float | None:
    """
    Perform the calculation based on the given operation.

    Takes two numbers and an operation symbol, performs the corresponding
    mathematical operation, and returns the result.

    Parameters
    ----------
    a: float
        The first operand.
    b: float
        The second operand.
    operation: str
        The operation to perform (+, -, *, /).

    Returns
    -------
    float | None
        The result of the calculation, or None if the operation is invalid
        or division by zero is attempted.

    Examples
    --------
    >>> calculate(5, 3, "+")
    8.0
    >>> calculate(5, 3, "-")
    2.0
    >>> calculate(5, 3, "*")
    15.0
    >>> calculate(5, 3, "/")
    1.6666666666666667
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


def display_result(result: float | None) -> None:
    """
    Display the result of the calculation.

    Parameters
    ----------
    result: float | None
        The result to display, or None if no result is available.
    """
    if result is not None:
        print(f"Result: {result}")


def main() -> None:
    """Executes the main functionality of the calculator."""
    print("Simple Calculator")
    print("Operations: +,-,*,/")

    a, b = get_numbers()
    operation = get_operation()
    result = calculate(a, b, operation)
    display_result(result)
    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
