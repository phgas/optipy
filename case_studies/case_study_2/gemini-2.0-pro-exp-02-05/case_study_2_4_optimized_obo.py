"""
User Manager
============

A simple module for managing user data.

This module provides functionalities to add, update, and retrieve user
information. It uses a JSON file to store user data and supports
operations such as adding a new user, updating existing user details,
and retrieving user information by name.

Examples
--------
>>> from user_manager import add_user, update_user, get_user
>>> users = []
>>> add_user(users, "John Doe", "30", "john.doe@example.com", "active")
>>> print(get_user(users, "John Doe"))
{'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com',
'status': 'active'}

>>> update_user(users, "John Doe", "age", "31")
>>> print(get_user(users, "John Doe"))
{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com',
'status': 'active'}
"""

import json
import logging
import os

import pyfiglet


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
    with an indent of 4 spaces.

    Parameters
    ----------
    data_to_save: list[dict]
        The data to be saved to the file.

    Returns
    -------
    None
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(users: list[dict], name: str, age: str, email: str, status: str) -> None:
    """
    Add a new user to the provided list and save it to the file.

    This function creates a new user dictionary with the given details,
    appends it to the list of users, saves the updated list to 'users.json',
    and logs an informational message.

    Parameters
    ----------
    users: list[dict]
        The list of existing user dictionaries.
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
    to the file.

    This function searches for a user by their name. If found, it updates
    the specified field with the new value, saves the updated list to
    'users.json', and logs an informational message. If the user is not
    found, it logs a message indicating this.

    Parameters
    ----------
    users: list[dict]
        The list of existing user dictionaries.
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.

    Returns
    -------
    None
    """
    for i, user in enumerate(users):
        if user["name"] == name:
            users[i][field] = value
            save_to_file(users)
            logging.info(f"Updated {name} {field}")
            return

    logging.info("User not found")


def get_user(users: list[dict], name: str) -> dict | str:
    """
    Retrieve a user from the provided list by their name.

    This function searches for a user by their name. If found, it returns
    the user dictionary. If not found, it returns the string 'Not found'.

    Parameters
    ----------
    users: list[dict]
        The list of existing user dictionaries.
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict | str
        The user dictionary if found, otherwise the string 'Not found'.
    """
    for user in users:
        if user["name"] == name:
            return user
    return "Not found"


def handle_add_user(users: list[dict]) -> None:
    """
    Handle the process of adding a new user by taking inputs.

    This function prompts for user details (name, age, email, and status),
    and then adds the new user to the provided list.

    Parameters
    ----------
    users: list[dict]
        The list of existing user dictionaries.

    Returns
    -------
    None
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    add_user(users, name, age, email, status)


def handle_update_user(users: list[dict]) -> None:
    """
    Handle the process of updating an existing user by taking inputs.

    This function prompts for the name of the user to update, the field to
    update, and the new value, then updates the user in the provided list.

    Parameters
    ----------
    users: list[dict]
        The list of existing user dictionaries.

    Returns
    -------
    None
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    update_user(users, name, field, value)


def handle_get_user(users: list[dict]) -> None:
    """
    Handle the process of retrieving and displaying a user's details.

    This function prompts for the name of the user to find, retrieves the
    user, and prints their details. If the user is not found, it logs a
    message.

    Parameters
    ----------
    users: list[dict]
        The list of existing user dictionaries.

    Returns
    -------
    None
    """
    name: str = input("Enter name to find: ")
    result: dict | str = get_user(users, name)
    if result == "Not found":
        logging.info("User not found")
    else:
        print(f"Name: {result['name']}")
        print(f"Age: {result['age']}")
        print(f"Email: {result['email']}")
        print(f"Status: {result['status']}")


def load_data_from_file(users: list[dict]) -> None:
    """
    Load user data from 'output.json' if it exists.

    This function checks if 'output.json' exists. If it does, it clears
    the provided user list and extends it with the data loaded from the
    file.

    Parameters
    ----------
    users: list[dict]
        The list of user dictionaries to be populated from the file.

    Returns
    -------
    None
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            users.clear()
            users.extend(json.load(file))


def display_menu() -> None:
    """
    Display the main menu options to the user.

    This function prints the available options: Add user, Update user,
    Get user, and Exit.

    Returns
    -------
    None
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")


def process_choice(choice: str, users: list[dict]) -> bool:
    """
    Process the user's choice and execute the corresponding action.

    This function takes the user's choice as input and calls the
    appropriate handler function based on the choice. It returns False if
    the choice is to exit, and True otherwise.

    Parameters
    ----------
    choice: str
        The user's menu choice.
    users: list[dict]
        The list of existing user dictionaries.

    Returns
    -------
    bool
        True if the program should continue running, False if it should
        exit.
    """
    if choice == "1":
        handle_add_user(users)
    elif choice == "2":
        handle_update_user(users)
    elif choice == "3":
        handle_get_user(users)
    elif choice == "4":
        print("Exiting...")
        return False
    else:
        print("Invalid choice, try again")
    return True


def main() -> None:
    """
    Execute the main functionality of the script.
    """
    users: list = []
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)

    load_data_from_file(users)

    while True:
        display_menu()
        choice: str = input("Enter choice: ")

        if not process_choice(choice, users):
            break


if __name__ == "__main__":
    main()
