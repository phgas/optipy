"""
User Management System
=====================

A simple user management system that allows adding, updating, and retrieving user information.

This module provides functionality to manage user data including creating, updating, and retrieving
user records. The user data is stored in a JSON file for persistence between program runs.

Examples
--------
>>> from user_manager import main
>>> main()  # Starts the interactive user management system

>>> from user_manager import add_user, get_user
>>> users = []
>>> users = add_user(users, "John Doe", "30", "john@example.com", "active")
>>> user = get_user(users, "John Doe")
>>> print(user)
{'name': 'John Doe', 'age': '30', 'email': 'john@example.com', 'status': 'active'}
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


def load_data() -> list[dict[str, str | int]]:
    """
    Load user data from the JSON file.

    Checks if the users.json file exists and loads the data if it does.
    If the file doesn't exist, returns an empty list.

    Returns
    -------
    list[dict[str, str | int]]
        A list of user dictionaries loaded from the file or an empty list.
    """
    if os.path.exists("users.json"):
        with open("users.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_to_file(users: list[dict[str, str | int]]) -> None:
    """
    Save user data to the JSON file.

    Writes the provided list of users to the users.json file in a formatted JSON structure.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The list of user dictionaries to save.
    """
    with open("users.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(users, indent=4))


def create_user(name: str, age: str, email: str, status: str) -> dict[str, str | int]:
    """
    Create a new user dictionary with the provided information.

    Parameters
    ----------
    name: str
        The name of the user.
    age: str
        The age of the user.
    email: str
        The email address of the user.
    status: str
        The status of the user (active/inactive).

    Returns
    -------
    dict[str, str | int]
        A dictionary containing the user information.
    """
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status
    }
    return user


def add_user(
    users: list[dict[str, str | int]], name: str, age: str, email: str, status: str
) -> list[dict[str, str | int]]:
    """
    Add a new user to the list and save to file.

    Creates a new user with the provided information, adds it to the list,
    saves the updated list to the file, and logs the action.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The current list of users.
    name: str
        The name of the user to add.
    age: str
        The age of the user to add.
    email: str
        The email address of the user to add.
    status: str
        The status of the user to add (active/inactive).

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of users.
    """
    user: dict[str, str | int] = create_user(name, age, email, status)
    users.append(user)
    save_to_file(users)
    logging.info(f"User added: {name}")
    return users


def update_user(
    users: list[dict[str, str | int]], name: str, field: str, value: str
) -> list[dict[str, str | int]]:
    """
    Update a specific field for a user.

    Finds the user with the given name and updates the specified field with the new value.
    Saves the updated list to the file and logs the action.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The current list of users.
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str
        The new value for the field.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of users.
    """
    for i, user in enumerate(users):
        if user["name"] == name:
            users[i][field] = value
            save_to_file(users)
            logging.info(f"Updated {name} {field}")
            return users

    logging.info("User not found")
    return users


def get_user(
    users: list[dict[str, str | int]], name: str
) -> dict[str, str | int] | None:
    """
    Find and return a user by name.

    Searches for a user with the given name in the list of users.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The list of users to search in.
    name: str
        The name of the user to find.

    Returns
    -------
    dict[str, str | int] | None
        The user dictionary if found, None otherwise.
    """
    for user in users:
        if user["name"] == name:
            return user
    return None


def display_user(user: dict[str, str | int] | None) -> None:
    """
    Display user information.

    Logs the user information if the user exists, otherwise logs that the user was not found.

    Parameters
    ----------
    user: dict[str, str | int] | None
        The user dictionary to display or None if the user was not found.
    """
    if user is None:
        logging.info("User not found")
        return

    logging.info(f"Name: {user['name']}")
    logging.info(f"Age: {user['age']}")
    logging.info(f"Email: {user['email']}")
    logging.info(f"Status: {user['status']}")


def process_add_user(
    users: list[dict[str, str | int]]
) -> list[dict[str, str | int]]:
    """
    Process the add user operation.

    Prompts the user for the necessary information to add a new user and calls the add_user function.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The current list of users.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of users.
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return add_user(users, name, age, email, status)


def process_update_user(
    users: list[dict[str, str | int]]
) -> list[dict[str, str | int]]:
    """
    Process the update user operation.

    Prompts the user for the necessary information to update a user and calls the update_user function.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The current list of users.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of users.
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return update_user(users, name, field, value)


def process_get_user(
    users: list[dict[str, str | int]]
) -> list[dict[str, str | int]]:
    """
    Process the get user operation.

    Prompts the user for the name of the user to find, calls the get_user function,
    and displays the user information.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The current list of users.

    Returns
    -------
    list[dict[str, str | int]]
        The unchanged list of users.
    """
    name: str = input("Enter name to find: ")
    user: dict[str, str | int] | None = get_user(users, name)
    display_user(user)
    return users


def display_menu() -> str:
    """
    Display the menu options and get the user's choice.

    Logs the available menu options and prompts the user for their choice.

    Returns
    -------
    str
        The user's choice as a string.
    """
    logging.info("1. Add user")
    logging.info("2. Update user")
    logging.info("3. Get user")
    logging.info("4. Exit")
    return input("Enter choice: ")


def process_user_choice(
    users: list[dict[str, str | int]], choice: str
) -> list[dict[str, str | int]] | None:
    """
    Process the user's menu choice.

    Calls the appropriate function based on the user's choice.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The current list of users.
    choice: str
        The user's menu choice.

    Returns
    -------
    list[dict[str, str | int]] | None
        The updated list of users or None if the user chose to exit.
    """
    if choice == "1":
        return process_add_user(users)
    elif choice == "2":
        return process_update_user(users)
    elif choice == "3":
        return process_get_user(users)
    elif choice == "4":
        logging.info("Exiting...")
        return None
    else:
        logging.info("Invalid choice, try again")
        return users


def display_header() -> None:
    """
    Display the application header.

    Uses pyfiglet to create and log a stylized header for the application.
    """
    header = pyfiglet.figlet_format("User Manager")
    logging.info(header)


def main() -> None:
    """
    Execute the main functionality of the script.

    Loads the user data, displays the header, and enters a loop to process user choices
    until the user chooses to exit.
    """
    users: list[dict[str, str | int]] = load_data()
    display_header()

    while True:
        choice: str = display_menu()
        users = process_user_choice(users, choice)
        if users is None:
            break


if __name__ == "__main__":
    main()
