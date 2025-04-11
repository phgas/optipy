"""
Simple Calculator
=================

A module that provides a basic calculator functionality.

This module implements a simple calculator that can perform basic arithmetic operations
such as addition, subtraction, multiplication, and division. It uses logging to track
operations and results, and handles errors like division by zero or invalid operations.

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


def get_input_number(prompt: str) -> float:
    """
    Prompt the user for a number and return it as a float.

    This function asks the user to input a number and converts it to a float.

    Parameters
    ----------
    prompt: str
        The message to display to the user when asking for input.

    Returns
    -------
    float
        The number entered by the user, converted to a float.

    Examples
    --------
    >>> get_input_number("Enter a number: ")
    Enter a number: 5
    5.0
    """
    return float(input(prompt))


def perform_calculation(number_a: float, number_b: float, operation: str) -> float | None:
    """
    Perform a calculation based on the given numbers and operation.

    This function performs the specified arithmetic operation on the two numbers.
    It handles division by zero and invalid operations by logging errors and returning None.

    Parameters
    ----------
    number_a: float
        The first number for the calculation.
    number_b: float
        The second number for the calculation.
    operation: str
        The arithmetic operation to perform ('+', '-', '*', or '/').

    Returns
    -------
    float | None
        The result of the calculation or None if an error occurs.

    Raises
    ------
    None

    Examples
    --------
    >>> perform_calculation(10, 5, '+')
    15.0
    >>> perform_calculation(10, 0, '/')
    [01-01-2023 12:00:00.000][ERROR][simple_calculator.py:1 - perform_calculation()]: Cannot divide by zero
    None
    """
    if operation == "+":
        return number_a + number_b
    elif operation == "-":
        return number_a - number_b
    elif operation == "*":
        return number_a * number_b
    elif operation == "/":
        if number_b == 0:
            logging.error("Cannot divide by zero")
            return None
        return number_a / number_b
    else:
        logging.error("Invalid operation")
        return None


def log_calculator_info() -> None:
    """
    Log information about the calculator.

    This function logs the name of the calculator and the supported operations.

    Parameters
    ----------
    None

    Returns
    -------
    None

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

    Parameters
    ----------
    None

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
    number_a: float = get_input_number("Enter first number: ")
    number_b: float = get_input_number("Enter second number: ")
    return number_a, number_b


def get_operation() -> str:
    """
    Get the desired mathematical operation from the user.

    This function prompts the user to enter a mathematical operation and returns it.

    Parameters
    ----------
    None

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

    Returns
    -------
    None

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

    This function logs a thank you message after the calculator has been used.

    Parameters
    ----------
    None

    Returns
    -------
    None

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
    to get numbers, perform calculations, and log results.

    Parameters
    ----------
    None

    Returns
    -------
    None

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
    number_a, number_b = get_numbers()
    operation = get_operation()
    result = perform_calculation(number_a, number_b, operation)
    log_result(result)
    log_thank_you()


if __name__ == "__main__":
    main()
