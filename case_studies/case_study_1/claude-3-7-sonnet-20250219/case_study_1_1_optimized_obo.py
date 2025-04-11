"""
Simple Calculator
================

A basic calculator module that performs arithmetic operations.

This module provides functions to get user input, perform calculations,
and display results. It supports addition, subtraction, multiplication,
and division operations.

Examples
--------
>>> from calculator import main
>>> main()
[INFO] Simple Calculator
[INFO] Operations: +,-,*,/
Enter first number: 5
Enter second number: 3
Enter operation (+, -, *, /): +
[INFO] Result: 8.0
[INFO] Thank you for using the calculator!
"""


# Standard library imports
import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def get_user_input() -> tuple[float, float, str]:
    """
    Prompt the user for two numbers and an operation and return them.

    This function collects user input for two numbers and an arithmetic
    operation to perform on them.

    Returns
    -------
    tuple[float, float, str]
        A tuple containing the first number, second number, and operation.

    Examples
    --------
    >>> get_user_input()  # User enters: 5, 3, +
    (5.0, 3.0, '+')
    """
    first_number: float = float(input("Enter first number: "))
    second_number: float = float(input("Enter second number: "))
    operation: str = input("Enter operation (+, -, *, /): ")
    return first_number, second_number, operation


def calculate(first_number: float, second_number: float,
              operation: str) -> float | None:
    """
    Perform the specified arithmetic operation on the given numbers.

    This function takes two numbers and performs the specified operation.
    It handles addition, subtraction, multiplication, and division.
    For division, it checks if the second number is zero to avoid division
    by zero.

    Parameters
    ----------
    first_number: float
        The first operand.
    second_number: float
        The second operand.
    operation: str
        The operation to perform (+, -, *, /).

    Returns
    -------
    float | None
        The result of the operation, or None if the operation is invalid
        or division by zero is attempted.

    Examples
    --------
    >>> calculate(5, 3, '+')
    8.0
    >>> calculate(5, 0, '/')
    None
    """
    if operation == "+":
        return first_number + second_number
    elif operation == "-":
        return first_number - second_number
    elif operation == "*":
        return first_number * second_number
    elif operation == "/":
        if second_number == 0:
            logging.error("Cannot divide by zero")
            return None
        return first_number / second_number
    else:
        logging.error("Invalid operation")
        return None


def main() -> None:
    """
    Execute the main functionality of the calculator script.

    This function displays information about the calculator, gets user input,
    performs the calculation, and displays the result.

    Examples
    --------
    >>> main()  # User enters: 5, 3, +
    [INFO] Simple Calculator
    [INFO] Operations: +,-,*,/
    [INFO] Result: 8.0
    [INFO] Thank you for using the calculator!
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")

    first_number: float
    second_number: float
    operation: str
    first_number, second_number, operation = get_user_input()
    result: float | None = calculate(first_number, second_number, operation)

    logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
