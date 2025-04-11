"""
User Management Module
======================

A module for managing user data, including adding, updating, retrieving,
and saving users to a file.

This module provides functions to manage user data, including adding new
users, updating existing users, retrieving user information, and saving
the user data to a JSON file. It also includes a simple command-line
interface for interacting with the user data.

Examples
--------
>>> from user_management import add_user, get_user
>>> add_user("John Doe", "30", "john.doe@example.com", "active")
>>> user = get_user("John Doe")
>>> print(user)
{'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com',
'status': 'active'}
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

users = []


def save_to_file(users_to_save: list[dict[str, str | int]]) -> None:
    """
    Save the list of users to a JSON file.

    This function writes the provided list of users to a file named
    "users.json" in JSON format.

    Parameters
    ----------
    users_to_save: list[dict[str, str | int]]
        A list of user dictionaries to be saved.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(users_to_save, indent=4))


def add_user(name: str, age: str, email: str, status: str) -> None:
    """
    Add a new user to the list and save it to the file.

    This function creates a new user dictionary with the provided details,
    appends it to the global users list, and saves the updated list to the
    file.

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
    """
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    users.append(user)
    save_to_file(users)
    logging.info(f"User added: {name}")


def update_user(name: str, field: str, value: str | int) -> None:
    """
    Update a user's information in the list and save it to the file.

    This function searches for a user by name, updates the specified field
    with the new value, and saves the updated list to the file. If the user
    is not found, it logs an info message.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str | int
        The new value for the specified field.
    """
    user_found: bool = False
    for i, user in enumerate(users):
        if user["name"] == name:
            users[i][field] = value
            user_found = True
    if not user_found:
        logging.info("User not found")
    else:
        save_to_file(users)
        logging.info(f"Updated {name} {field}")


def get_user(name: str) -> dict[str, str | int] | str:
    """
    Retrieve a user's information by name.

    This function searches for a user by name in the global users list and
    returns the user's information if found. If the user is not found, it
    returns "Not found".

    Parameters
    ----------
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict[str, str | int] | str
        The user's information as a dictionary if found, otherwise
        "Not found".
    """
    for user in users:
        if user["name"] == name:
            return user
    return "Not found"


def load_users() -> None:
    """
    Load users from a JSON file into the global users list.

    This function reads user data from a file named "output.json" and
    populates the global users list. If the file does not exist, it does
    nothing.
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            users.clear()
            users.extend(json.load(file))


def display_header() -> None:
    """
    Display the application header using ASCII art.

    This function uses the pyfiglet library to generate and print an ASCII
    art header for the application.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)


def display_menu() -> None:
    """
    Display the main menu options to the user.

    This function prints the available options for the user to interact
    with the application.
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")


def handle_choice(choice: str) -> bool:
    """
    Handle the user's menu choice and perform the corresponding action.

    This function processes the user's input choice, performs the
    corresponding action, and returns a boolean indicating whether to
    continue the application loop.

    Parameters
    ----------
    choice: str
        The user's menu choice.

    Returns
    -------
    bool
        True if the application should continue, False if it should exit.
    """
    if choice == "1":
        name: str = input("Enter name: ")
        age: str = input("Enter age: ")
        email: str = input("Enter email: ")
        status: str = input("Enter status (active/inactive): ")
        add_user(name, age, email, status)
    elif choice == "2":
        name: str = input("Enter name to update: ")
        field: str = input("Enter field to update (name/age/email/status): ")
        value: str = input("Enter new value: ")
        update_user(name, field, value)
    elif choice == "3":
        name: str = input("Enter name to find: ")
        result = get_user(name)
        if result == "Not found":
            logging.info("User not found")
        else:
            print(f"Name: {result['name']}")
            print(f"Age: {result['age']}")
            print(f"Email: {result['email']}")
            print(f"Status: {result['status']}")
    elif choice == "4":
        logging.info("Exiting...")
        return False
    else:
        logging.info("Invalid choice, try again")
    return True


def main() -> None:
    """Executes the main functionality of the script."""
    display_header()
    while True:
        display_menu()
        choice: str = input("Enter choice: ")
        if not handle_choice(choice):
            break
        load_users()


if __name__ == "__main__":
    main()
