"""
User Manager
============

A simple module for managing users.

This module provides functionalities to add, update, and retrieve user
information. It uses a JSON file to store user data and supports basic
operations like adding a new user, updating existing user details, and
retrieving user information by name.

Examples
--------
>>> from user_manager import add_user, update_user, get_user
>>> users = []
>>> add_user(users, "John Doe", "30", "john.doe@example.com", "active")
>>> update_user(users, "John Doe", "age", "31")
>>> user = get_user(users, "John Doe")
>>> print(user)
{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com',
'status': 'active'}
"""

import json
import logging
import os

import pyfiglet  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def save_to_file(data_to_save: list[dict]) -> None:
    """
    Save the provided data to a JSON file named 'users.json'.

    This function serializes the given data and writes it to 'users.json'
    with an indentation of 4 spaces.

    Parameters
    ----------
    data_to_save: list[dict]
        The data to be saved to the file.

    Returns
    -------
    None

    Examples
    --------
    >>> save_to_file([{'name': 'John Doe', 'age': '30'}])
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(users: list[dict], name: str, age: str, email: str, status: str) -> None:
    """
    Add a new user to the provided list and save it to a file.

    This function creates a new user dictionary and appends it to the list
    of users. After adding the user, it saves the updated list to
    'users.json'.

    Parameters
    ----------
    users: list[dict]
        The list of existing users.
    name: str
        The name of the new user.
    age: str
        The age of the new user.
    email: str
        The email address of the new user.
    status: str
        The status of the new user (e.g., 'active' or 'inactive').

    Returns
    -------
    None

    Examples
    --------
    >>> users = []
    >>> add_user(users, "John Doe", "30", "john.doe@example.com", "active")
    """
    user: dict = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    users.append(user)
    save_to_file(users)
    logging.info(f"User added: {name}")


def update_user(users: list[dict], name: str, field: str, value: str) -> None:
    """
    Update the specified field of a user in the provided list and save it
    to a file.

    This function searches for a user by name and updates the specified
    field with the new value. If the user is found and updated, it saves
    the updated list to 'users.json'.

    Parameters
    ----------
    users: list[dict]
        The list of existing users.
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.

    Returns
    -------
    None

    Examples
    --------
    >>> users = [{'name': 'John Doe', 'age': '30'}]
    >>> update_user(users, "John Doe", "age", "31")
    """
    for index, user in enumerate(users):
        if user["name"] == name:
            users[index][field] = value
            save_to_file(users)
            logging.info(f"Updated {name} {field}")
            return

    logging.info("User not found")


def get_user(users: list[dict], name: str) -> dict | str:
    """
    Retrieve a user from the provided list by name.

    This function searches for a user by name and returns the user
    dictionary if found. If the user is not found, it returns the string
    'Not found'.

    Parameters
    ----------
    users: list[dict]
        The list of existing users.
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict | str
        The user dictionary if found, otherwise the string 'Not found'.

    Examples
    --------
    >>> users = [{'name': 'John Doe', 'age': '30'}]
    >>> user = get_user(users, "John Doe")
    >>> print(user)
    {'name': 'John Doe', 'age': '30'}

    >>> user = get_user(users, "Jane Doe")
    >>> print(user)
    Not found
    """
    for user in users:
        if user["name"] == name:
            return user
    return "Not found"


def handle_add_user(users: list[dict]) -> None:
    """
    Handle the process of adding a new user by taking inputs.

    This function prompts the user to enter details for a new user and then
    adds the user.

    Parameters
    ----------
    users: list[dict]
        The list of existing users.

    Returns
    -------
    None

    Examples
    --------
    >>> users = []
    >>> handle_add_user(users)
    Enter name: John Doe
    Enter age: 30
    Enter email: john.doe@example.com
    Enter status (active/inactive): active
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    add_user(users, name, age, email, status)


def handle_update_user(users: list[dict]) -> None:
    """
    Handle the process of updating an existing user by taking inputs.

    This function prompts the user to enter details for updating an
    existing user and then updates the user.

    Parameters
    ----------
    users: list[dict]
        The list of existing users.

    Returns
    -------
    None

    Examples
    --------
    >>> users = [{'name': 'John Doe', 'age': '30'}]
    >>> handle_update_user(users)
    Enter name to update: John Doe
    Enter field to update (name/age/email/status): age
    Enter new value: 31
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    update_user(users, name, field, value)


def handle_get_user(users: list[dict]) -> None:
    """
    Handle the process of retrieving and displaying a user by taking inputs.

    This function prompts the user to enter the name of the user to find
    and then displays the user's details.

    Parameters
    ----------
    users: list[dict]
        The list of existing users.

    Returns
    -------
    None

    Examples
    --------
    >>> users = [{'name': 'John Doe', 'age': '30', 'email':
    'john@example.com', 'status': 'active'}]
    >>> handle_get_user(users)
    Enter name to find: John Doe
    Name: John Doe
    Age: 30
    Email: john@example.com
    Status: active
    """
    name: str = input("Enter name to find: ")
    result: dict | str = get_user(users, name)
    if result == "Not found":
        logging.info("User not found")
    else:
        logging.info(f"Name: {result['name']}")
        logging.info(f"Age: {result['age']}")
        logging.info(f"Email: {result['email']}")
        logging.info(f"Status: {result['status']}")


def load_data_from_file(users: list[dict]) -> None:
    """
    Load user data from 'output.json' if it exists.

    This function checks if 'output.json' exists and, if so, loads the data
    into the provided users list.

    Parameters
    ----------
    users: list[dict]
        The list to which the loaded users will be added.

    Returns
    -------
    None

    Examples
    --------
    >>> users = []
    >>> load_data_from_file(users)
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            users.clear()
            users.extend(json.load(file))


def display_menu() -> None:
    """
    Display the main menu options.

    This function prints the available options to the console.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> display_menu()
    1. Add user
    2. Update user
    3. Get user
    4. Exit
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")


def handle_user_choice(choice: str, users: list[dict]) -> bool:
    """
    Handle the user's choice from the main menu.

    This function takes the user's choice and performs the corresponding
    action.

    Parameters
    ----------
    choice: str
        The user's choice from the main menu.
    users: list[dict]
        The list of existing users.

    Returns
    -------
    bool
        True if the program should continue running, False if it should exit.

    Examples
    --------
    >>> users = []
    >>> handle_user_choice("1", users)
    Enter name: John Doe
    Enter age: 30
    Enter email: john.doe@example.com
    Enter status (active/inactive): active
    True

    >>> handle_user_choice("4", users)
    Exiting...
    False
    """
    if choice == "1":
        handle_add_user(users)
    elif choice == "2":
        handle_update_user(users)
    elif choice == "3":
        handle_get_user(users)
    elif choice == "4":
        logging.info("Exiting...")
        return False
    else:
        logging.info("Invalid choice, try again")
    return True


def main() -> None:
    """
    Execute the main functionality of the script.

    This function initializes the user list, displays the main menu,
    and handles user input until the user chooses to exit.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    users = []

    while True:
        display_menu()
        choice: str = input("Enter choice: ")

        if not handle_user_choice(choice, users):
            break

        load_data_from_file(users)


if __name__ == "__main__":
    main()
