"""
Simple Calculator
================

A basic calculator that performs addition, subtraction, multiplication, and division.

This module provides a simple command-line calculator that takes two numbers and
an operation as input, then displays the result of the calculation.

Examples
--------
>>> python calculator.py
Simple Calculator
Operations: +,-,*,/
Enter first number: 5
Enter second number: 3
Enter operation: +
Result: 8.0
Thank you for using the calculator!
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


def get_numbers() -> tuple[float, float]:
    """
    Get two numbers from the user.

    Prompts the user to enter two numbers that will be used as operands
    in a mathematical operation.

    Returns
    -------
    tuple[float, float]
        A tuple containing the two numbers entered by the user.
    """
    first_number = float(input("Enter first number: "))
    second_number = float(input("Enter second number: "))
    return first_number, second_number


def get_operation() -> str:
    """
    Get the desired operation from the user.

    Prompts the user to enter the mathematical operation they want to perform.

    Returns
    -------
    str
        The operation symbol entered by the user (+, -, *, or /).
    """
    return input("Enter operation: ")


def calculate(first_number: float, second_number: float, operation: str) -> float:
    """
    Perform the calculation based on the given numbers and operation.

    Takes two numbers and an operation symbol, then performs the corresponding
    mathematical operation.

    Parameters
    ----------
    first_number: float
        The first operand.
    second_number: float
        The second operand.
    operation: str
        The operation to perform (+, -, *, or /).

    Returns
    -------
    float
        The result of the calculation.

    Raises
    ------
    ValueError
        If division by zero is attempted or an invalid operation is provided.
    """
    if operation == "+":
        return first_number + second_number
    elif operation == "-":
        return first_number - second_number
    elif operation == "*":
        return first_number * second_number
    elif operation == "/":
        if second_number == 0:
            raise ValueError("Cannot divide by zero")
        return first_number / second_number
    else:
        raise ValueError("Invalid operation")


def display_result(result: float) -> None:
    """
    Display the result of the calculation.

    Parameters
    ----------
    result: float
        The result to display.
    """
    print(f"Result: {result}")


def main() -> None:
    """Executes the main functionality of the calculator."""
    print("Simple Calculator")
    print("Operations: +,-,*,/")

    first_number, second_number = get_numbers()
    operation = get_operation()

    try:
        result = calculate(first_number, second_number, operation)
        display_result(result)
    except ValueError as e:
        logging.error(f"Calculation error: {e}")
        print(e)

    print("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
