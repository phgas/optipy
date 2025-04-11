"""
Simple calculator
=================

A simple calculator module that performs basic arithmetic operations.

This module allows users to perform addition, subtraction, multiplication, and division.
It prompts the user for two numbers and an operation, then displays the result.

Examples
--------
>>> from simple_calculator import calculate
>>> calculate(10, 5, "+")
15.0

>>> calculate(10, 5, "*")
50.0

>>> calculate(10, 0, "/")
Cannot divide by zero.

>>> calculate(10, 5, "$")
Invalid operation.
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


def calculate(first_operand: float, second_operand: float, operation: str) -> float | None:
    """
    Perform a calculation based on the given operation and return the result.

    This function takes two numbers and an operation, performs the calculation,
    and returns the result. It handles division by zero and invalid operations.

    Parameters
    ----------
    first_operand: float
        The first number.
    second_operand: float
        The second number.
    operation: str
        The operation to perform (+, -, *, /).

    Returns
    -------
    float | None
        The result of the calculation, or None if the operation is invalid.

    Raises
    ------
    ValueError
        If attempting to divide by zero.

    Examples
    --------
    >>> calculate(10, 5, "+")
    15.0

    >>> calculate(10, 5, "*")
    50.0
    """
    if operation == "+":
        result = first_operand + second_operand
    elif operation == "-":
        result = first_operand - second_operand
    elif operation == "*":
        result = first_operand * second_operand
    elif operation == "/":
        if second_operand == 0:
            logging.error("Cannot divide by zero.")
            raise ValueError("Cannot divide by zero.")
        result = first_operand / second_operand
    else:
        logging.error("Invalid operation.")
        return None
    return result


def get_input() -> tuple[float, float, str]:
    """
    Get inputs from the user and return them as a tuple.

    This function prompts the user for two numbers and an operation,
    and returns the inputs.

    Returns
    -------
    tuple[float, float, str]
        A tuple containing the first number, second number, and the operation.

    Examples
    --------
    (Assuming user inputs 10, 5, and "+")
    >>> get_input()
    (10.0, 5.0, '+')
    """
    first_operand = float(
        input(
            "Enter first number that will be used as the first operand in the"
            " mathematical operation that you want to perform: "
        )
    )
    second_operand = float(
        input(
            "Enter second number that will be used as the second operand in the"
            " mathematical operation that you want to perform: "
        )
    )
    operation = input(
        "Enter the desired mathematical operation that you would like to perform on"
        " the two numbers you just provided (+, -, *, /): "
    )
    return first_operand, second_operand, operation


def main() -> None:
    """Executes the main functionality of the script."""
    logging.info("Simple Calculator")
    logging.info("Operations: +, -, *, /")

    first_operand, second_operand, operation = get_input()
    try:
        result = calculate(first_operand, second_operand, operation)
        if result is not None:
            logging.info(f"Result: {result}")
    except ValueError:
        pass
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
