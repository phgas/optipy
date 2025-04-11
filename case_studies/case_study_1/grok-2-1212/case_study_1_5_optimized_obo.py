"""
Simple Calculator
================

A module that provides a basic calculator functionality.

This module implements a simple calculator that can perform basic arithmetic operations
such as addition, subtraction, multiplication, and division. It uses logging to provide
informative messages about the operations and results.

Examples
--------
>>> from simple_calculator import main
>>> main()
[12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Simple Calculator
[12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Operations: +,-,*,/
[12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - process_result()]: Result: 7.0
[12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - process_result()]: Thank you for using the calculator!
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


def get_input(prompt: str) -> float:
    """
    Prompt the user for input and convert it to a float.

    This function asks the user for a numerical input and converts it to a float.
    It is used to get numbers for the calculator operations.

    Parameters
    ----------
    prompt: str
        The message to display to the user when asking for input.

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


def perform_operation(a: float, b: float, op: str) -> float | None:
    """
    Perform the specified arithmetic operation on two numbers.

    This function takes two numbers and an operation, then performs the operation.
    It handles addition, subtraction, multiplication, and division, with special
    handling for division by zero and invalid operations.

    Parameters
    ----------
    a: float
        The first number for the operation.
    b: float
        The second number for the operation.
    op: str
        The operation to perform ('+', '-', '*', or '/').

    Returns
    -------
    float | None
        The result of the operation, or None if the operation is invalid or division by zero occurs.

    Raises
    ------
    logging.error
        Logs an error message if division by zero or an invalid operation is attempted.

    Examples
    --------
    >>> perform_operation(5, 3, '+')
    8.0
    >>> perform_operation(5, 0, '/')
    [12-05-2023 14:30:00.000][ERROR][simple_calculator.py:1 - perform_operation()]: Cannot divide by zero
    None
    """
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            logging.error("Cannot divide by zero")
            return None
        return a / b
    else:
        logging.error("Invalid operation")
        return None


def log_calculator_info() -> None:
    """
    Log information about the calculator.

    This function logs the name of the calculator and the supported operations.

    Examples
    --------
    >>> log_calculator_info()
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Simple Calculator
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Operations: +,-,*,/
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")


def get_numbers() -> tuple[float, float]:
    """
    Get two numbers from the user.

    This function prompts the user to enter two numbers and returns them as a tuple.

    Returns
    -------
    tuple[float, float]
        A tuple containing the two numbers entered by the user.

    Examples
    --------
    >>> get_numbers()
    Enter first number: 5
    Enter second number: 3
    (5.0, 3.0)
    """
    first_number: float = get_input("Enter first number: ")
    second_number: float = get_input("Enter second number: ")
    return first_number, second_number


def get_operation() -> str:
    """
    Get the desired mathematical operation from the user.

    This function prompts the user to enter a mathematical operation and returns it.

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
    return input("Enter the desired mathematical operation (+, -, *, /): ")


def process_result(result: float | None) -> None:
    """
    Process and log the result of a calculation.

    This function logs the result of a calculation if it is not None, and logs a thank you message.

    Parameters
    ----------
    result: float | None
        The result of the calculation to be processed.

    Examples
    --------
    >>> process_result(7.0)
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - process_result()]: Result: 7.0
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - process_result()]: Thank you for using the calculator!
    """
    if result is not None:
        logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


def main() -> None:
    """
    Execute the main functionality of the calculator.

    This function orchestrates the calculator's operations by logging information,
    getting numbers and an operation from the user, performing the operation, and
    processing the result.

    Examples
    --------
    >>> main()
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Simple Calculator
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Operations: +,-,*,/
    Enter first number: 5
    Enter second number: 3
    Enter the desired mathematical operation (+, -, *, /): +
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - process_result()]: Result: 8.0
    [12-05-2023 14:30:00.000][INFO][simple_calculator.py:1 - process_result()]: Thank you for using the calculator!
    """
    log_calculator_info()
    first_number, second_number = get_numbers()
    operation = get_operation()
    result = perform_operation(first_number, second_number, operation)
    process_result(result)


if __name__ == "__main__":
    main()
