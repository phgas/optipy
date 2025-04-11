"""
Simple calculator
=================

A module that provides basic arithmetic operations.

This module allows users to perform addition, subtraction, multiplication, and division
on two numbers. It handles division by zero and invalid operation inputs.

Examples
--------
>>> from simple_calculator import calculate
>>> calculate(10, 5, "+")
Result: 15.0

>>> calculate(10, 5, "-")
Result: 5.0

>>> calculate(10, 5, "*")
Result: 50.0

>>> calculate(10, 5, "/")
Result: 2.0

>>> calculate(10, 0, "/")
Cannot divide by zero.
Result: 0

>>> calculate(10, 5, "&")
Invalid operation.
Result: 0
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


def get_number(prompt: str) -> float:
    """
    Get a valid number from the user and return it.

    This function continuously prompts the user for input until a valid
    floating-point number is entered.

    Parameters
    ----------
    prompt: str
        The message displayed to the user.

    Returns
    -------
    float
        The valid number entered by the user.

    Examples
    --------
    >>> get_number("Enter first number: ")
    Enter first number: 10
    10.0
    """
    while True:
        try:
            number: float = float(input(prompt))
            return number
        except ValueError:
            logging.error("Invalid input. Please enter a valid number.")


def get_operation() -> str:
    """
    Get a valid mathematical operation from the user and return it.

    This function prompts the user to enter a mathematical operation
    (+, -, *, /) and validates the input.

    Returns
    -------
    str
        The valid operation entered by the user.

    Examples
    --------
    >>> get_operation()
    Enter the desired mathematical operation (+, -, *, /): +
    '+'
    """
    while True:
        operation: str = input(
            "Enter the desired mathematical operation (+, -, *, /): "
        )
        is_valid_operation = operation in ["+", "-", "*", "/"]
        if is_valid_operation:
            return operation
        else:
            logging.error("Invalid operation. Please enter +, -, *, or /.")


def calculate(first_number: float, second_number: float, operation: str) -> float:
    """
    Perform the specified mathematical operation on two numbers and return the result.

    This function takes two numbers and an operation, then performs the
    calculation. It handles division by zero and invalid operations.

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
    float
        The result of the calculation.

    Examples
    --------
    >>> calculate(10, 5, "+")
    15.0

    >>> calculate(10, 0, "/")
    Cannot divide by zero.
    0.0
    """
    result: float = 0.0
    if operation == "+":
        result = first_number + second_number
    elif operation == "-":
        result = first_number - second_number
    elif operation == "*":
        result = first_number * second_number
    elif operation == "/":
        if second_number == 0:
            logging.error("Cannot divide by zero.")
        else:
            result = first_number / second_number
    else:
        logging.error("Invalid operation.")
    return result


def main() -> None:
    """Executes the main functionality of the script."""
    logging.info("Simple Calculator")
    logging.info("Operations: +, -, *, /")

    first_number: float = get_number(
        "Enter first number for the mathematical operation: "
    )
    second_number: float = get_number(
        "Enter second number for the mathematical operation: "
    )
    operation: str = get_operation()

    result: float = calculate(first_number, second_number, operation)
    logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
