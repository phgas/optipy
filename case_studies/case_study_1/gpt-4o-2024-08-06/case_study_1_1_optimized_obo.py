"""
Simple Calculator
=================

A simple calculator module that performs basic arithmetic operations.

This module provides functionality to perform addition, subtraction, 
multiplication, and division based on user input. It logs the operations 
and handles division by zero and invalid operations gracefully.

Examples
--------
>>> from simple_calculator import main
>>> main()
Enter first number: 10
Enter second number: 5
Enter the desired operation (+, -, *, /): +
[Result: 15.0]
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


def get_inputs() -> tuple[float, float, str]:
    """
    Prompt the user for two numbers and an arithmetic operation.

    This function collects user input for two numbers and an operation 
    to be performed on them.

    Returns
    -------
    tuple[float, float, str]
        A tuple containing the first number, second number, and the 
        operation.
    """
    first_number: float = float(input("Enter first number: "))
    second_number: float = float(input("Enter second number: "))
    operation: str = input("Enter the desired operation (+, -, *, /): ")
    return first_number, second_number, operation


def calculate(
    first_number: float, second_number: float, operation: str
) -> float | None:
    """
    Perform the specified arithmetic operation on two numbers.

    This function takes two numbers and an operation, performs the 
    operation, and returns the result. It logs errors for division by 
    zero and invalid operations.

    Parameters
    ----------
    first_number: float
        The first number for the operation.
    second_number: float
        The second number for the operation.
    operation: str
        The arithmetic operation to perform.

    Returns
    -------
    float | None
        The result of the operation, or None if an error occurs.
    """
    if operation == "+":
        return first_number + second_number
    elif operation == "-":
        return first_number - second_number
    elif operation == "*":
        return first_number * second_number
    elif operation == "/":
        if second_number == 0:
            logging.error("Cannot divide by zero")
            return None
        return first_number / second_number
    else:
        logging.error("Invalid operation")
        return None


def main() -> None:
    """Execute the main functionality of the script."""
    logging.info("Simple Calculator")
    logging.info("Operations: +,-,*,/")
    first_number, second_number, operation = get_inputs()
    result = calculate(first_number, second_number, operation)
    if result is not None:
        logging.info(f"Result: {result}")
    logging.info("Thank you for using the calculator!")


if __name__ == "__main__":
    main()
