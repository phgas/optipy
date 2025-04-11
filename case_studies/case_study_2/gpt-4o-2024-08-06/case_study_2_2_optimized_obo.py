"""
User Management Module
======================

This module provides functionalities to manage user data, including
adding, updating, retrieving, and saving user information. It also
provides a command-line interface for interacting with the user data.

Examples
--------
>>> from user_management import main
>>> main()
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
    Load user data from a JSON file and return it as a list of dictionaries.

    If the file does not exist, return an empty list.

    Returns
    -------
    list[dict[str, str | int]]
        A list of user data dictionaries.
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def save_to_file(data: list[dict[str, str | int]]) -> None:
    """
    Save the given user data to a JSON file.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of user data dictionaries to be saved.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4))


def add_user(
    data: list[dict[str, str | int]], name: str, age: str, email: str, status: str
) -> None:
    """
    Add a new user to the data list and save it to the file.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of user data dictionaries.
    name: str
        The name of the user.
    age: str
        The age of the user.
    email: str
        The email of the user.
    status: str
        The status of the user (active/inactive).
    """
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    data.append(user)
    save_to_file(data)
    logging.info(f"User added: {name}")


def update_user(
    data: list[dict[str, str | int]], name: str, field: str, value: str
) -> None:
    """
    Update a specific field of a user in the data list and save it to the file.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of user data dictionaries.
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str
        The new value for the specified field.
    """
    for user in data:
        if user["name"] == name:
            user[field] = value
            save_to_file(data)
            logging.info(f"Updated {name} {field}")
            return
    logging.info("User not found")


def get_user(
    data: list[dict[str, str | int]], name: str
) -> dict[str, str | int] | str:
    """
    Retrieve a user's data by name from the data list.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of user data dictionaries.
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict[str, str | int] | str
        The user's data dictionary if found, otherwise "Not found".
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def display_header() -> None:
    """
    Display the header for the user management system.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)


def get_user_choice() -> str:
    """
    Display the menu options and get the user's choice.

    Returns
    -------
    str
        The user's choice as a string.
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")
    return input("Enter choice: ")


def handle_choice(choice: str, data: list[dict[str, str | int]]) -> bool:
    """
    Handle the user's choice and perform the corresponding action.

    Parameters
    ----------
    choice: str
        The user's choice.
    data: list[dict[str, str | int]]
        A list of user data dictionaries.

    Returns
    -------
    bool
        False if the user chooses to exit, otherwise True.
    """
    if choice == "1":
        name: str = input("Enter name: ")
        age: str = input("Enter age: ")
        email: str = input("Enter email: ")
        status: str = input("Enter status (active/inactive): ")
        add_user(data, name, age, email, status)
    elif choice == "2":
        name: str = input("Enter name to update: ")
        field: str = input("Enter field to update (name/age/email/status): ")
        value: str = input("Enter new value: ")
        update_user(data, name, field, value)
    elif choice == "3":
        name: str = input("Enter name to find: ")
        result = get_user(data, name)
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
    data = load_data()
    while True:
        choice = get_user_choice()
        if not handle_choice(choice, data):
            break


if __name__ == "__main__":
    main()
