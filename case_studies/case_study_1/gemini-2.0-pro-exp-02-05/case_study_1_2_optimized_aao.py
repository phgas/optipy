"""
Simple calculator
=================

A module that provides basic arithmetic operations.

This module allows users to perform addition, subtraction, multiplication, and division.
It prompts the user for two numbers and an operation, then displays the result.

Examples
--------
>>> Enter first number that will be used as the first operand in the
mathematical operation that you want to perform: 10
>>> Enter second number that will be used as the second operand in the
mathematical operation that you want to perform: 5
>>> Enter the desired mathematical operation that you would like to perform on
the two numbers you just provided (+, -, *, /): +
>>> Result: 15.0
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
    Get a valid number from the user.

    This function prompts the user to enter a number until a valid float is
    provided.

    Parameters
    ----------
    prompt: str
        The message displayed to the user.

    Returns
    -------
    float
        The number entered by the user.

    Examples
    --------
    >>> get_number("Enter a number: ")
    Enter a number: 10
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
    Get a valid operation from the user.

    This function prompts the user to enter an operation until a valid
    operation (+, -, *, /) is provided.

    Returns
    -------
    str
        The operation entered by the user.

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
        if operation in ["+", "-", "*", "/"]:
            return operation
        else:
            logging.error("Invalid operation. Please enter +, -, *, or /.")


def calculate(first_number: float, second_number: float, operation: str) -> float | None:
    """
    Perform the calculation based on the given operation.

    This function takes two numbers and an operation, then performs the
    calculation. It returns the result or None if the operation is invalid.

    Parameters
    ----------
    first_number: float
        The first number.
    second_number: float
        The second number.
    operation: str
        The operation to perform.

    Returns
    -------
    float | None
        The result of the calculation, or None if the operation is invalid.

    Raises
    ------
    ValueError
        If division by zero is attempted.

    Examples
    --------
    >>> calculate(10.0, 5.0, "+")
    15.0
    >>> calculate(10.0, 5.0, "/")
    2.0
    """
    if operation == "+":
        result: float = first_number + second_number
    elif operation == "-":
        result: float = first_number - second_number
    elif operation == "*":
        result: float = first_number * second_number
    elif operation == "/":
        if second_number == 0:
            logging.error("Cannot divide by zero.")
            return None
        result: float = first_number / second_number
    else:
        logging.error("Invalid operation.")
        return None
    return result


def main() -> None:
    """Executes the main functionality of the script."""
    logging.info("Simple Calculator")
    logging.info("Operations: +, -, *, /")

    first_number: float = get_number(
        "Enter first number that will be used as the first operand in the "
        "mathematical operation that you want to perform: "
    )
    second_number: float = get_number(
        "Enter second number that will be used as the second operand in the "
        "mathematical operation that you want to perform: "
    )
    operation: str = get_operation()

    result: float | None = calculate(first_number, second_number, operation)

    if result is not None:
        logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
