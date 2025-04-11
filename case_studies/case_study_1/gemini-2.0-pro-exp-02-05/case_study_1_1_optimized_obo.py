"""
Calculator
==========

A simple calculator module for performing basic arithmetic operations.

This module provides functions to perform addition, subtraction,
multiplication, and division.
It also includes input validation and logging of operations and errors.

Examples
--------
>>> from calculator import calculate, get_operands_and_operation
>>> first_operand, second_operand, operation = get_operands_and_operation()
Enter first number that will be used as the first operand in the
mathematical operation that you want to perform: 5
Enter second number that will be used as the second operand in the
mathematical operation that you want to perform: 0
Enter the desired mathematical operation that you would like to perform on
the two numbers you just provided (+, -, *, /): /
>>> result = calculate(first_operand, second_operand, operation)
Cannot divide by zero
>>> print(result)
None

>>> from calculator import calculate, get_operands_and_operation
>>> first_operand, second_operand, operation = get_operands_and_operation()
Enter first number that will be used as the first operand in the
mathematical operation that you want to perform: 5
Enter second number that will be used as the second operand in the
mathematical operation that you want to perform: 5
Enter the desired mathematical operation that you would like to perform on
the two numbers you just provided (+, -, *, /): +
>>> result = calculate(first_operand, second_operand, operation)
>>> print(result)
10.0
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
    Get the input from the user and return it as a float.

    This function prompts the user with the provided message and retrieves
    the input.
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


def calculate(
    first_operand: float, second_operand: float, operation: str
) -> float | None:
    """
    Calculate the result of the operation on the two operands and return
    the result.

    This function takes two operands and an operation, performs the
    calculation, and returns the result. It handles division by zero and
    invalid operations.

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
    float | None
        The result of the calculation, or None if the operation is invalid
        or division by zero occurs.

    Examples
    --------
    >>> calculate(5, 5, "+")
    10.0
    >>> calculate(5, 0, "/")
    Cannot divide by zero
    None
    >>> calculate(5, 5, "invalid")
    Invalid operation
    None
    """
    if operation == "+":
        return first_operand + second_operand
    if operation == "-":
        return first_operand - second_operand
    if operation == "*":
        return first_operand * second_operand
    if operation == "/":
        division_by_zero: bool = second_operand == 0
        if division_by_zero:
            logging.info("Cannot divide by zero")
            return None
        return first_operand / second_operand
    logging.info("Invalid operation")
    return None


def get_operands_and_operation() -> tuple[float, float, str]:
    """
    Get the operands and operation from the user and return them.

    This function prompts the user to enter two numbers and an operation.
    It retrieves the inputs and returns them as a tuple.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[float, float, str]
        A tuple containing the first operand, second operand, and the
        operation.

    Examples
    --------
    >>> get_operands_and_operation()
    Enter first number that will be used as the first operand in the
    mathematical operation that you want to perform: 5
    Enter second number that will be used as the second operand in the
    mathematical operation that you want to perform: 5
    Enter the desired mathematical operation that you would like to perform
    on the two numbers you just provided (+, -, *, /): +
    (5.0, 5.0, '+')
    """
    first_operand: float = get_input(
        "Enter first number that will be used as the first operand in the"
        " mathematical operation that you want to perform: "
    )
    second_operand: float = get_input(
        "Enter second number that will be used as the second operand in the"
        " mathematical operation that you want to perform: "
    )
    operation: str = input(
        "Enter the desired mathematical operation that you would like to"
        " perform on the two numbers you just provided (+, -, *, /): "
    )
    return first_operand, second_operand, operation


def display_result(result: float | None) -> None:
    """
    Display the result of the calculation.

    This function takes the result of a calculation and displays it.
    If the result is None, it does nothing.

    Parameters
    ----------
    result: float | None
        The result of the calculation.

    Returns
    -------
    None

    Examples
    --------
    >>> display_result(10.0)
    Result: 10.0
    >>> display_result(None)
    """
    if result is not None:
        logging.info(f"Result: {result}")


def main() -> None:
    """
    Execute the main functionality of the script.
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")

    first_operand: float
    second_operand: float
    operation: str
    first_operand, second_operand, operation = get_operands_and_operation()
    result: float | None = calculate(first_operand, second_operand, operation)
    display_result(result)

    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
