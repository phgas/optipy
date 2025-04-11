"""
Calculator module
================

A simple calculator module for performing basic arithmetic operations.

This module provides functions to get user input, perform calculations,
and display results.
It supports addition, subtraction, multiplication, and division.

Examples
--------
>>> from calculator import perform_calculation
>>> perform_calculation()
Enter first number that will be used as the first operand in the
mathematical operation that you want to perform: 5
Enter second number that will be used as the second operand in the
mathematical operation that you want to perform: 5
Enter the desired mathematical operation that you would like to perform
on the two numbers you just provided (+, -, *, /): +
[<date>][INFO][calculator.py:<line_number> - perform_calculation()]:
Result: 10.0
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


def get_input_value(prompt: str) -> float:
    """
    Get a float input from the user with a given prompt and return it.

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
    >>> get_input_value("Enter a number: ")
    Enter a number: 5
    5.0
    """
    return float(input(prompt))


def get_operation() -> str:
    """
    Get the desired mathematical operation from the user and return it.

    Returns
    -------
    str
        The operation entered by the user.

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
    first_number: float, second_number: float, operation: str
) -> float | None:
    """
    Calculate the result of the operation on the two numbers and return it.

    Parameters
    ----------
    first_number: float
        The first number.
    second_number: float
        The second number.
    operation: str
        The operation to perform.

    Returns
    -------
    float | None
        The result of the calculation, or None if the operation is
        invalid or division by zero occurs.

    Examples
    --------
    >>> calculate(5, 5, '+')
    10.0
    >>> calculate(5, 5, '-')
    0.0
    >>> calculate(5, 5, '*')
    25.0
    >>> calculate(5, 5, '/')
    1.0
    >>> calculate(5, 0, '/')
    None
    >>> calculate(5, 5, 'invalid')
    None
    """
    if operation == "+":
        return first_number + second_number
    elif operation == "-":
        return first_number - second_number
    elif operation == "*":
        return first_number * second_number
    elif operation == "/":
        if second_number == 0:
            logging.info("Cannot divide by zero")
            return None
        else:
            return first_number / second_number
    else:
        logging.info("Invalid operation")
        return None


def perform_calculation() -> None:
    """
    Get input from the user, perform the calculation, and display the
    result.

    Parameters
    ----------

    Returns
    -------

    Raises
    ------

    Examples
    --------
    >>> perform_calculation()
    Enter first number that will be used as the first operand in the
    mathematical operation that you want to perform: 5
    Enter second number that will be used as the second operand in the
    mathematical operation that you want to perform: 5
    Enter the desired mathematical operation that you would like to
    perform on the two numbers you just provided (+, -, *, /): +
    [<date>][INFO][calculator.py:<line_number> - perform_calculation()]:
    Result: 10.0
    """
    first_number: float = get_input_value(
        "Enter first number that will be used as the first operand in the "
        "mathematical operation that you want to perform: "
    )
    second_number: float = get_input_value(
        "Enter second number that will be used as the second operand in the "
        "mathematical operation that you want to perform: "
    )
    operation: str = get_operation()

    result: float | None = calculate(first_number, second_number, operation)

    if result is not None:
        logging.info(f"Result: {result}")


def main() -> None:
    """
    Execute the main functionality of the script.

    Parameters
    ----------

    Returns
    -------

    Raises
    ------

    Examples
    --------
    >>> main()
    [<date>][INFO][calculator.py:<line_number> - main()]: Simple Calculator
    [<date>][INFO][calculator.py:<line_number> - main()]: Operations:
    +,-,*,/
    Enter first number that will be used as the first operand in the
    mathematical operation that you want to perform: 5
    Enter second number that will be used as the second operand in the
    mathematical operation that you want to perform: 5
    Enter the desired mathematical operation that you would like to
    perform on the two numbers you just provided (+, -, *, /): +
    [<date>][INFO][calculator.py:<line_number> - perform_calculation()]:
    Result: 10.0
    [<date>][INFO][calculator.py:<line_number> - main()]: Thank you for
    using the calculator!
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")

    perform_calculation()

    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
