"""
Calculator
==========

This module provides a basic calculator functionality.

It allows users to perform simple arithmetic operations such as
addition, subtraction, multiplication, and division.
The module prompts the user for two numbers and an operation, then
displays the result.

Examples
--------
>>> Enter first number that will be used as the first operand in the
mathematical operation that you want to perform: 5
>>> Enter second number that will be used as the second operand in the
mathematical operation that you want to perform: 5
>>> Enter the desired mathematical operation that you would like to
perform on the two numbers you just provided (+, -, *, /):
first_number=5.0, second_number=5.0*
>>> Result: result=25.0
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


def get_input_value(prompt: str) -> float:
    """
    Get a float input from the user via the given prompt and return it.

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


def get_operator(first_number: float, second_number: float) -> str:
    """
    Get the mathematical operator from the user and return it.

    Parameters
    ----------
    first_number: float
        The first number entered by the user.
    second_number: float
        The second number entered by the user.

    Returns
    -------
    str
        The operator entered by the user.

    Examples
    --------
    >>> get_operator(5.0, 3.0)
    Enter the desired mathematical operation that you would like to
    perform on the two numbers you just provided (+, -, *, /):
    first_number=5.0, second_number=3.0+
    '+'
    """
    return input(
        "Enter the desired mathematical operation that you would like to "
        "perform on the two numbers you just provided (+, -, *, /): "
        f"{first_number=}, {second_number=}"
    )


def calculate_result(
    first_number: float, second_number: float, operator: str
) -> float | None:
    """
    Calculate the result of the operation on the two numbers and return it.

    If the operation is invalid or a division by zero occurs, log an error
    and return None.

    Parameters
    ----------
    first_number: float
        The first number.
    second_number: float
        The second number.
    operator: str
        The operator.

    Returns
    -------
    float | None
        The result of the operation, or None if an error occurred.

    Examples
    --------
    >>> calculate_result(5.0, 3.0, '+')
    8.0
    >>> calculate_result(5.0, 0.0, '/')
    ERROR:root:Cannot divide by zero
    None
    >>> calculate_result(5.0, 3.0, '&')
    ERROR:root:Invalid operation
    None
    """
    if operator == "+":
        return first_number + second_number
    if operator == "-":
        return first_number - second_number
    if operator == "*":
        return first_number * second_number
    if operator == "/":
        if second_number == 0:
            logging.error("Cannot divide by zero")
            return None
        return first_number / second_number
    logging.error("Invalid operation")
    return None


def display_result(result: float | None) -> None:
    """
    Display the result of the calculation.

    If the result is not None, log the result.

    Parameters
    ----------
    result: float | None
        The result of the calculation.

    Examples
    --------
    >>> display_result(8.0)
    INFO:root:Result: result=8.0
    >>> display_result(None)
    """
    if result is not None:
        logging.info(f"Result: {result=}")


def calculator_setup() -> None:
    """
    Set up the calculator by logging the initial messages.

    Examples
    --------
    >>> calculator_setup()
    INFO:root:Simple Calculator
    INFO:root:Operations: +,-,*,/
    """
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")


def get_inputs() -> tuple[float, float, str]:
    """
    Get the inputs (two numbers and an operator) from the user and return
    them.

    Returns
    -------
    tuple[float, float, str]
        A tuple containing the first number, second number, and the
        operator.

    Examples
    --------
    >>> get_inputs()
    Enter first number that will be used as the first operand in the
    mathematical operation that you want to perform: 5
    Enter second number that will be used as the second operand in the
    mathematical operation that you want to perform: 5
    Enter the desired mathematical operation that you would like to
    perform on the two numbers you just provided (+, -, *, /):
    first_number=5.0, second_number=5.0*
    (5.0, 5.0, '*')
    """
    first_number: float = get_input_value(
        "Enter first number that will be used as the first operand in the"
        " mathematical operation that you want to perform: "
    )
    second_number: float = get_input_value(
        "Enter second number that will be used as the second operand in the"
        " mathematical operation that you want to perform: "
    )
    operator: str = get_operator(first_number, second_number)
    return first_number, second_number, operator


def main() -> None:
    """
    Execute the main functionality of the script.

    This function sets up the calculator, gets inputs from the user,
    calculates the result, displays the result, and logs a thank you
    message.
    """
    calculator_setup()
    first_number: float
    second_number: float
    operator: str
    first_number, second_number, operator = get_inputs()
    result: float | None = calculate_result(
        first_number, second_number, operator)
    display_result(result)
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
