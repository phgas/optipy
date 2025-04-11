"""
Simple calculator
=================

A simple calculator module that performs basic arithmetic operations.

This module allows users to perform addition, subtraction, multiplication, and division
on two numbers. It handles division by zero and invalid operation inputs.

Examples
--------
>>> from simple_calculator import calculate
>>> result = calculate(10, 5, "+")
>>> print(result)
15.0

>>> result = calculate(10, 5, "/")
>>> print(result)
2.0

>>> result = calculate(10, 0, "/")
Cannot divide by zero.
0

>>> result = calculate(10, 5, "$")
Invalid operation.
0
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


def calculate(first_number: float, second_number: float, operation: str) -> float:
    """
    Perform a mathematical operation on two numbers and return the result.

    This function takes two numbers and an operation as input, performs the
    specified operation, and returns the result. It handles division by zero
    and invalid operation inputs.

    Parameters
    ----------
    first_number: float
        The first number.
    second_number: float
        The second number.
    operation: str
        The operation to perform ("+", "-", "*", "/").

    Returns
    -------
    float
        The result of the operation.

    Examples
    --------
    >>> calculate(10, 5, "+")
    15.0
    >>> calculate(10, 5, "-")
    5.0
    >>> calculate(10, 5, "*")
    50.0
    >>> calculate(10, 5, "/")
    2.0
    """
    result: float = 0
    if operation == "+":
        result = first_number + second_number
    elif operation == "-":
        result = first_number - second_number
    elif operation == "*":
        result = first_number * second_number
    elif operation == "/":
        if second_number == 0:
            logging.info("Cannot divide by zero.")
        else:
            result = first_number / second_number
    else:
        logging.info("Invalid operation.")
    return result


def get_input(prompt: str) -> float:
    """
    Get a float input from the user and return it.

    This function prompts the user for input and ensures that the input is a
    valid float. It continues to prompt the user until a valid float is
    entered.

    Parameters
    ----------
    prompt: str
        The prompt message to display to the user.

    Returns
    -------
    float
        The float value entered by the user.

    Examples
    --------
    >>> get_input("Enter a number: ")
    Enter a number: 10
    10.0
    """
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            logging.info(
                "Invalid input. Please enter a valid number (int/float).")


def get_operation() -> str:
    """
    Get a valid operation input from the user and return it.

    This function prompts the user to enter a mathematical operation and
    ensures that the input is one of the valid operations (+, -, *, /). It
    continues to prompt the user until a valid operation is entered.

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
        operation = input(
            "Enter the desired mathematical operation (+, -, *, /): "
        )
        if operation in ("+", "-", "*", "/"):
            return operation
        logging.info("Invalid operation. Please enter a valid operation.")


def main() -> None:
    """
    Execute the main functionality of the simple calculator.

    This function gets the inputs from the user, performs the calculation,
    and prints the result.
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +, -, *, /")

    first_number = get_input(
        "Enter first number: "
    )
    second_number = get_input(
        "Enter second number: "
    )
    operation = get_operation()

    result = calculate(first_number, second_number, operation)
    logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
