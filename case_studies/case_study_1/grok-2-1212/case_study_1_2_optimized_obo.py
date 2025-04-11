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
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>]: Simple Calculator
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>]: Operations: +,-,*,/
Enter first number: 10
Enter second number: 5
Enter the desired mathematical operation (+, -, *, /): +
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>]: Result: 15.0
[01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - <module>]: Thank you for using the calculator!
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


def get_user_input(prompt: str) -> float:
    """
    Prompt the user for input and convert it to a float.

    This function asks the user for a numerical input and converts it to a float.
    It is used to get numbers for calculations.

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
    >>> get_user_input("Enter a number: ")
    Enter a number: 5
    5.0
    """
    return float(input(prompt))


def perform_calculation(num1: float, num2: float, operation: str) -> float | None:
    """
    Perform a calculation based on the given numbers and operation.

    This function performs the specified arithmetic operation on the two given numbers.
    It handles addition, subtraction, multiplication, and division, logging errors for
    invalid operations or division by zero.

    Parameters
    ----------
    num1: float
        The first number for the calculation.
    num2: float
        The second number for the calculation.
    operation: str
        The arithmetic operation to perform ('+', '-', '*', or '/').

    Returns
    -------
    float | None
        The result of the calculation or None if an error occurs.

    Raises
    ------
    logging.error
        If the operation is invalid or if attempting to divide by zero.

    Examples
    --------
    >>> perform_calculation(10, 5, '+')
    15.0
    >>> perform_calculation(10, 0, '/')
    [01-01-2023 12:00:00.000][ERROR][simple_calculator.py:1 - perform_calculation()]: Cannot divide by zero
    None
    """
    if operation == "+":
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        if num2 == 0:
            logging.error("Cannot divide by zero")
            return None
        return num1 / num2
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
    num1: float = get_user_input("Enter first number: ")
    num2: float = get_user_input("Enter second number: ")
    return num1, num2


def get_operation() -> str:
    """
    Get the desired mathematical operation from the user.

    This function prompts the user to enter a mathematical operation and returns it.

    Returns
    -------
    str
        The mathematical operation entered by the user.

    Examples
    --------
    >>> get_operation()
    Enter the desired mathematical operation (+, -, *, /): +
    '+'
    """
    return input("Enter the desired mathematical operation (+, -, *, /): ")


def log_result(result: float | None) -> None:
    """
    Log the result of a calculation.

    This function logs the result of a calculation if it is not None.

    Parameters
    ----------
    result: float | None
        The result of the calculation to be logged.

    Examples
    --------
    >>> log_result(15.0)
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - log_result()]: Result: 15.0
    """
    if result is not None:
        logging.info(f"Result: {result}")


def log_thank_you() -> None:
    """
    Log a thank you message.

    This function logs a thank you message to the user after using the calculator.

    Examples
    --------
    >>> log_thank_you()
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - log_thank_you()]: Thank you for using the calculator!
    """
    logging.info("Thank you for using the calculator!")


def main() -> None:
    """
    Execute the main functionality of the calculator.

    This function orchestrates the calculator's operations by logging information,
    getting user input for numbers and operations, performing the calculation, and
    logging the result and a thank you message.

    Examples
    --------
    >>> main()
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Simple Calculator
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Operations: +,-,*,/
    Enter first number: 10
    Enter second number: 5
    Enter the desired mathematical operation (+, -, *, /): +
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Result: 15.0
    [01-01-2023 12:00:00.000][INFO][simple_calculator.py:1 - main()]: Thank you for using the calculator!
    """
    log_calculator_info()
    num1, num2 = get_numbers()
    operation = get_operation()
    result = perform_calculation(num1, num2, operation)
    log_result(result)
    log_thank_you()


if __name__ == "__main__":
    main()
