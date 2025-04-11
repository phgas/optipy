"""
Simple Calculator
================

A module that provides a basic calculator functionality.

This module implements a simple calculator that can perform basic arithmetic operations
such as addition, subtraction, multiplication, and division. It uses logging to track
operations and results.

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


def get_input(prompt: str) -> float:
    """
    Prompt the user for input and convert it to a float.

    This function asks the user for a numeric input and converts it to a float.

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

    This function performs basic arithmetic operations based on the provided operator.
    It logs errors for invalid operations or division by zero.

    Parameters
    ----------
    a: float
        The first number.
    b: float
        The second number.
    op: str
        The operation to perform ('+', '-', '*', or '/').

    Returns
    -------
    float | None
        The result of the operation, or None if the operation is invalid or division by zero occurs.

    Raises
    ------
    logging.error
        If the operation is invalid or if attempting to divide by zero.

    Examples
    --------
    >>> perform_operation(10, 5, '+')
    15.0
    >>> perform_operation(10, 0, '/')
    [01-01-2023 12:00:00.000][ERROR][simple_calculator.py:1 - perform_operation()]: Cannot divide by zero
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
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Simple Calculator
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - log_calculator_info()]: Operations: +,-,*,/
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
    Enter first number: 10
    Enter second number: 5
    (10.0, 5.0)
    """
    a: float = get_input("Enter first number: ")
    b: float = get_input("Enter second number: ")
    return a, b


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


def calculate_and_log_result(a: float, b: float, op: str) -> None:
    """
    Calculate the result of an operation and log it.

    This function performs the specified operation on the given numbers and logs the result.

    Parameters
    ----------
    a: float
        The first number.
    b: float
        The second number.
    op: str
        The operation to perform.

    Examples
    --------
    >>> calculate_and_log_result(10, 5, '+')
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - calculate_and_log_result()]: Result: 15.0
    """
    result: float | None = perform_operation(a, b, op)
    if result is not None:
        logging.info(f"Result: {result}")


def log_thank_you() -> None:
    """
    Log a thank you message.

    This function logs a message thanking the user for using the calculator.

    Examples
    --------
    >>> log_thank_you()
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - log_thank_you()]: Thank you for using the calculator!
    """
    logging.info("Thank you for using the calculator!")


def main() -> None:
    """
    Execute the main functionality of the calculator.

    This function orchestrates the calculator's operations by calling other functions
    to get numbers, perform operations, and log results.

    Examples
    --------
    >>> main()
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Simple Calculator
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Operations: +,-,*,/
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Enter first number: 10
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Enter second number: 5
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Enter the desired mathematical operation (+, -, *, /): +
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Result: 15.0
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Thank you for using the calculator!
    """
    log_calculator_info()
    a, b = get_numbers()
    op = get_operation()
    calculate_and_log_result(a, b, op)
    log_thank_you()


if __name__ == "__main__":
    main()
