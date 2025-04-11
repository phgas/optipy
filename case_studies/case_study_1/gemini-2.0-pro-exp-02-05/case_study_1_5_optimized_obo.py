"""
Calculator module
=================

A simple calculator module for performing basic arithmetic operations.

This module provides functions to get user input, perform calculations,
and display results.
It supports addition, subtraction, multiplication, and division.

Examples
--------
>>> from calculator import run_calculator
>>> run_calculator()
Enter first number that will be used as the first operand in the
mathematical operation that you want to perform: 5
Enter second number that will be used as the second operand in the
mathematical operation that you want to perform: 5
Enter the desired mathematical operation that you would like to perform
on the two numbers you just provided (+, -, *, /): +
[03-03-2024 14:55:17.610][INFO][calculator.py:63 - display_result()]:
Result: 10.0
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


def get_input(prompt: str) -> float:
    """
    Get input from the user and return it as a float.

    This function prompts the user with the provided message and
    retrieves input.
    It converts the input to a float and returns it.

    Parameters
    ----------
    prompt: str
        The message to display to the user.

    Returns
    -------
    float
        The user's input converted to a float.

    Examples
    --------
    >>> get_input("Enter a number: ")
    Enter a number: 5
    5.0
    """
    return float(input(prompt))


def get_operation() -> str:
    """
    Get the desired mathematical operation from the user.

    This function prompts the user to enter the desired mathematical
    operation.
    It accepts '+', '-', '*', or '/' as valid operations.

    Returns
    -------
    str
        The mathematical operation entered by the user.

    Examples
    --------
    >>> get_operation()
    Enter the desired mathematical operation that you would like to
    perform on the two numbers you just provided (+, -, *, /): +
    '+'
    """
    return input(
        "Enter the desired mathematical operation that you would like to "
        "perform on the two numbers you just provided (+, -, *, /): "
    )


def calculate(
    first_operand: float, second_operand: float, operation: str
) -> float | str:
    """
    Calculate the result of the specified operation on two operands.

    This function takes two operands and an operation as input.
    It performs the calculation based on the given operation and returns
    the result.
    If the operation is invalid or division by zero occurs, it returns an
    error message.

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
    >>> calculate(5, 3, '+')
    8

    >>> calculate(5, 0, '/')
    'Cannot divide by zero'

    >>> calculate(5, 3, '$')
    'Invalid operation'
    """
    if operation == "+":
        return first_operand + second_operand
    if operation == "-":
        return first_operand - second_operand
    if operation == "*":
        return first_operand * second_operand
    if operation == "/":
        if second_operand == 0:
            return "Cannot divide by zero"
        return first_operand / second_operand
    return "Invalid operation"


def display_result(result: float | str) -> None:
    """
    Display the result of the calculation.

    This function takes the result of a calculation as input.
    If the result is a string (indicating an error), it logs an error
    message.
    Otherwise, it logs the result as an informational message.

    Parameters
    ----------
    result: float | str
        The result of the calculation or an error message.

    Examples
    --------
    >>> display_result(10.5)
    INFO:root:Result: 10.5

    >>> display_result("Cannot divide by zero")
    ERROR:root:Cannot divide by zero
    """
    if isinstance(result, str):
        logging.error(result)
    else:
        logging.info(f"Result: {result}")


def run_calculator() -> None:
    """
    Run the calculator application.

    This function prompts the user for two operands and the desired
    operation.
    It then performs the calculation and displays the result.

    Examples
    --------
    >>> run_calculator()
    Enter first number that will be used as the first operand in the
    mathematical operation that you want to perform: 5
    Enter second number that will be used as the second operand in the
    mathematical operation that you want to perform: 3
    Enter the desired mathematical operation that you would like to
    perform on the two numbers you just provided (+, -, *, /): +
    INFO:root:Result: 8.0
    """
    first_operand: float = get_input(
        "Enter first number that will be used as the first operand in the"
        " mathematical operation that you want to perform: "
    )
    second_operand: float = get_input(
        "Enter second number that will be used as the second operand in the"
        " mathematical operation that you want to perform: "
    )
    operation: str = get_operation()

    result: float | str = calculate(first_operand, second_operand, operation)
    display_result(result)


def main() -> None:
    """
    Executes the main functionality of the script.

    This function initializes the calculator, logs the start and end of
    the program,
    and calls the run_calculator function to execute the calculator logic.
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")

    run_calculator()

    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
