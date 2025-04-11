"""
Calculator module
=================

A simple calculator module for performing basic arithmetic operations.

This module provides functions to get user input for numbers and
operations, calculate the result of the operation, and display the
result.

Examples
--------
>>> from calculator import get_float_input, get_operation_input, \
calculate_result, display_result
>>> first_number = get_float_input("Enter the first number: ")
Enter the first number: 5
>>> second_number = get_float_input("Enter the second number: ")
Enter the second number: 6
>>> operation = get_operation_input()
Enter the desired mathematical operation that you would like to \
perform on the two numbers you just provided (+, -, *, /): +
>>> result = calculate_result(first_number, second_number, operation)
>>> display_result(result)
[01-01-2024 12:00:00.000][INFO][calculator.py:93 - \
display_result()]: Result: 11.0
[01-01-2024 12:00:00.000][INFO][calculator.py:94 - \
display_result()]: Thank you for using the calculator!
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


def get_float_input(prompt: str) -> float:
    """
    Get float input from the user and return it.

    This function prompts the user to enter a number and returns the
    input as a float.

    Parameters
    ----------
    prompt: str
        The message that will be displayed to the user.

    Returns
    -------
    float
        The given input of the user as a float.

    Examples
    --------
    >>> get_float_input("Enter a number: ")
    Enter a number: 5
    5.0
    """
    return float(input(prompt))


def get_operation_input() -> str:
    """
    Get the desired mathematical operation from the user and return it.

    This function prompts the user to enter a mathematical operation
    (+, -, *, /) and returns their input.

    Returns
    -------
    str
        The given mathematical operation of the user.

    Examples
    --------
    >>> get_operation_input()
    Enter the desired mathematical operation that you would like to \
perform on the two numbers you just provided (+, -, *, /): +
    '+'
    """
    return input(
        "Enter the desired mathematical operation that you would like "
        "to perform on the two numbers you just provided (+, -, *, /): "
    )


def calculate_result(
    first_operand: float, second_operand: float, operation: str
) -> float | str:
    """
    Calculate the result of the given operation on the two operands and
    return it.

    This function takes two operands and an operation, performs the
    calculation, and returns the result. If the operation is invalid
    or division by zero occurs, it returns an error message.

    Parameters
    ----------
    first_operand: float
        The first operand.
    second_operand: float
        The second operand.
    operation: str
        The operation to perform (+, -, *, /).

    Returns
    -------
    float | str
        The result of the calculation or an error message.

    Examples
    --------
    >>> calculate_result(5, 3, '+')
    8

    >>> calculate_result(5, 0, '/')
    'Cannot divide by zero'

    >>> calculate_result(5, 3, '$')
    'Invalid operation'
    """
    if operation == "+":
        return first_operand + second_operand
    elif operation == "-":
        return first_operand - second_operand
    elif operation == "*":
        return first_operand * second_operand
    elif operation == "/":
        if second_operand == 0:
            return "Cannot divide by zero"
        else:
            return first_operand / second_operand
    else:
        return "Invalid operation"


def display_result(result: float | str) -> None:
    """
    Display the result of the calculation.

    This function logs the result of the calculation and a thank you
    message.

    Parameters
    ----------
    result: float | str
        The result of the calculation or an error message.

    Examples
    --------
    >>> display_result(15)
    [01-01-2024 12:00:00.000][INFO][calculator.py:132 - \
display_result()]: Result: 15
    [01-01-2024 12:00:00.000][INFO][calculator.py:133 - \
display_result()]: Thank you for using the calculator!

    >>> display_result("Cannot divide by zero")
    [01-01-2024 12:00:00.000][INFO][calculator.py:132 - \
display_result()]: Result: Cannot divide by zero
    [01-01-2024 12:00:00.000][INFO][calculator.py:133 - \
display_result()]: Thank you for using the calculator!
    """
    logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


def main() -> None:
    """
    Execute the main functionality of the script.

    This function serves as the entry point for the calculator program.
    It prompts the user for two numbers and an operation, performs the
    calculation, and displays the result.
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")

    first_operand: float = get_float_input(
        "Enter first number that will be used as the first operand in "
        "the mathematical operation that you want to perform: "
    )
    second_operand: float = get_float_input(
        "Enter second number that will be used as the second operand in "
        "the mathematical operation that you want to perform: "
    )
    operation: str = get_operation_input()

    result: float | str = calculate_result(
        first_operand, second_operand, operation
    )
    display_result(result)


if __name__ == "__main__":
    main()
