"""
User manager
============

A simple module for managing users.

This module provides functionalities to add, update, and retrieve user
information. It uses a JSON file to store user data.

Examples
--------
>>> from user_manager import add_user, update_user, get_user
>>> add_user("John Doe", "30", "john.doe@example.com", "active")
User added: John Doe

>>> update_user("John Doe", "age", "31")
Updated John Doe age

>>> get_user("John Doe")
{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com', 'status': 'active'}
"""


# Standard library imports
import json
import logging
import os

# Third party imports
import pyfiglet  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def save_to_file(data_to_save: list[dict[str, str]]) -> None:
    """
    Save the given data to a JSON file.

    Parameters
    ----------
    data_to_save: list[dict[str, str]]
        The data to be saved.

    Examples
    --------
    >>> save_to_file([{'name': 'John Doe', 'age': '30'}])
    """
    with open("users.json", "w", encoding="utf-8") as file:
        json.dump(data_to_save, file, indent=4)


def add_user(
    name: str, age: str, email: str, status: str, data: list[dict[str, str]]
) -> None:
    """
    Add a new user to the data list and save it to the file.

    Parameters
    ----------
    name: str
        The name of the user.
    age: str
        The age of the user.
    email: str
        The email of the user.
    status: str
        The status of the user.
    data: list[dict[str, str]]
        The list to which the user will be appended.

    Examples
    --------
    >>> data = []
    >>> add_user("John Doe", "30", "john@example.com", "active", data)
    """
    user: dict[str, str] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    data.append(user)
    save_to_file(data)
    logging.info(f"User added: {name}")


def update_user(
    name: str, field: str, value: str, data: list[dict[str, str]]
) -> None:
    """
    Update user information based on name.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update.
    value: str
        The new value for the field.
    data: list[dict[str, str]]
        The list in which the user will be updated.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30'}]
    >>> update_user("John Doe", "age", "31", data)
    """
    user_was_found: bool = False
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            user_was_found = True
    if not user_was_found:
        logging.warning(f"User not found: {name}")
    else:
        save_to_file(data)
        logging.info(f"Updated {name} {field}")


def get_user(name: str, data: list[dict[str, str]]) -> dict[str, str] | str:
    """
    Find and return a user.

    Parameters
    ----------
    name: str
        The name of the user to find.
    data: list[dict[str, str]]
        The list in which the user will be searched for.

    Returns
    -------
    dict[str, str] | str
        The user dictionary if found, "Not found" otherwise.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30'}]
    >>> get_user("John Doe", data)
    {'name': 'John Doe', 'age': '30'}

    >>> get_user("Jane Doe", data)
    'Not found'
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str]]:
    """
    Load user data from the JSON file if it exists.

    Returns
    -------
    list[dict[str, str]]
        The loaded user data.

    Examples
    --------
    >>> load_data()
    [{'name': 'John Doe', 'age': '30'}]
    """
    data: list[dict[str, str]] = []
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    return data


def get_input(prompt: str) -> str:
    """
    Get user input.

    Parameters
    ----------
    prompt: str
        The prompt message.

    Returns
    -------
    str
        The user input.

    Examples
    --------
    >>> get_input("Enter your name: ")
    """
    return input(prompt)


def process_choice(choice: str, data: list[dict[str, str]]) -> bool:
    """
    Process the user's choice and perform the corresponding action.

    Parameters
    ----------
    choice: str
        The user's choice.
    data: list[dict[str, str]]
        The current user data.

    Returns
    -------
    bool
        True if the program should continue, False if it should exit.

    Examples
    --------
    >>> data = []
    >>> process_choice("1", data)
    True
    """
    if choice == "1":
        name: str = get_input("Enter name: ")
        age: str = get_input("Enter age: ")
        email: str = get_input("Enter email: ")
        status: str = get_input("Enter status (active/inactive): ")
        add_user(name, age, email, status, data)
    elif choice == "2":
        name = get_input("Enter name to update: ")
        field = get_input("Enter field to update (name/age/email/status): ")
        value = get_input("Enter new value: ")
        update_user(name, field, value, data)
    elif choice == "3":
        name = get_input("Enter name to find: ")
        result: dict[str, str] | str = get_user(name, data)
        if result == "Not found":
            logging.warning(f"User not found: {name}")
        else:
            logging.info(f"Name: {result['name']}")
            logging.info(f"Age: {result['age']}")
            logging.info(f"Email: {result['email']}")
            logging.info(f"Status: {result['status']}")
    elif choice == "4":
        logging.info("Exiting...")
        return False
    else:
        logging.warning("Invalid choice, try again.")
    return True


def main() -> None:
    """Executes the main functionality of the script."""
    header: str = pyfiglet.figlet_format("User Manager")
    logging.info(header)
    data: list[dict[str, str]] = load_data()
    while True:
        logging.info("1. Add user")
        logging.info("2. Update user")
        logging.info("3. Get user")
        logging.info("4. Exit")
        choice: str = get_input("Enter choice: ")

        should_continue: bool = process_choice(choice, data)
        if not should_continue:
            break


if __name__ == "__main__":
    main()
