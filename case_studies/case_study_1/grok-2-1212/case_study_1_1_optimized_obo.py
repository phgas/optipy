"""
Simple Calculator
================

A module that provides a basic calculator functionality.

This module includes functions to get user input, perform basic arithmetic operations,
and log the results. It supports addition, subtraction, multiplication, and division.

Examples
--------
>>> from simple_calculator import main
>>> main()
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Simple Calculator
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Operations: +,-,*,/
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Enter first number: 10
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Enter second number: 5
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Enter the desired mathematical operation (+, -, *, /): +
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Result: 15.0
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Thank you for using the calculator!
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


def get_float_input(prompt: str) -> float:
    """
    Prompts the user for a float input and returns it.

    This function uses the built-in input function to get a string from the user,
    converts it to a float, and returns the result.

    Parameters
    ----------
    prompt: str
        The message to display to the user when asking for input.

    Returns
    -------
    float
        The float value entered by the user.

    Examples
    --------
    >>> get_float_input("Enter a number: ")
    10.5
    """
    return float(input(prompt))


def perform_calculation(a: float, b: float, operation: str) -> float | None:
    """
    Performs a mathematical operation on two numbers and returns the result.

    This function supports addition, subtraction, multiplication, and division.
    If an invalid operation is provided or if division by zero is attempted,
    it logs an error and returns None.

    Parameters
    ----------
    a: float
        The first number for the operation.
    b: float
        The second number for the operation.
    operation: str
        The operation to perform ('+', '-', '*', or '/').

    Returns
    -------
    float | None
        The result of the operation or None if an error occurs.

    Raises
    ------
    logging.error
        If the operation is invalid or if division by zero is attempted.

    Examples
    --------
    >>> perform_calculation(10, 5, '+')
    15.0
    >>> perform_calculation(10, 0, '/')
    None
    """
    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        if b == 0:
            logging.error("Cannot divide by zero")
            return None
        return a / b
    else:
        logging.error("Invalid operation")
        return None


def log_calculator_info() -> None:
    """
    Logs information about the calculator.

    This function logs the name of the calculator and the supported operations.

    Examples
    --------
    >>> log_calculator_info()
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Simple Calculator
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Operations: +,-,*,/
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")


def get_user_input() -> tuple[float, float, str]:
    """
    Collects user input for two numbers and an operation.

    This function prompts the user to enter two numbers and an operation,
    then returns them as a tuple.

    Returns
    -------
    tuple[float, float, str]
        A tuple containing the first number, the second number, and the operation.

    Examples
    --------
    >>> first_number, second_number, operation = get_user_input()
    Enter first number: 10
    Enter second number: 5
    Enter the desired mathematical operation (+, -, *, /): +
    >>> first_number
    10.0
    >>> second_number
    5.0
    >>> operation
    '+'
    """
    first_number: float = get_float_input("Enter first number: ")
    second_number: float = get_float_input("Enter second number: ")
    operation: str = input(
        "Enter the desired mathematical operation (+, -, *, /): ")
    return first_number, second_number, operation


def process_result(result: float | None) -> None:
    """
    Processes and logs the result of a calculation.

    This function logs the result if it is not None and logs a thank you message.

    Parameters
    ----------
    result: float | None
        The result of the calculation to be processed.

    Examples
    --------
    >>> process_result(15.0)
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Result: 15.0
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Thank you for using the calculator!
    >>> process_result(None)
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Thank you for using the calculator!
    """
    if result is not None:
        logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


def main() -> None:
    """
    Executes the main functionality of the calculator.

    This function orchestrates the calculator's operations by logging information,
    getting user input, performing calculations, and processing the results.

    Examples
    --------
    >>> main()
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Simple Calculator
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Operations: +,-,*,/
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Enter first number: 10
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Enter second number: 5
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Enter the desired mathematical operation (+, -, *, /): +
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Result: 15.0
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>()]: Thank you for using the calculator!
    """
    log_calculator_info()
    first_number: float
    second_number: float
    operation: str
    first_number, second_number, operation = get_user_input()
    result: float | None = perform_calculation(
        first_number, second_number, operation)
    process_result(result)


if __name__ == "__main__":
    main()
