"""
User Management Module
======================

This module provides functionalities to manage user data, including
adding, updating, retrieving, and saving user information. It also
provides a command-line interface for user interaction.

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


def save_to_file(data: list[dict[str, str | int]]) -> None:
    """
    Save the provided data to a JSON file.

    This function writes the given list of user dictionaries to a file
    named "users.json".

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of user dictionaries to be saved.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4))


def add_user(
    data: list[dict[str, str | int]], name: str, age: str, email: str, status: str
) -> None:
    """
    Add a new user to the data list and save it to the file.

    This function creates a new user dictionary with the provided
    details, appends it to the data list, and saves the updated list
    to the file.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        The list of existing user dictionaries.
    name: str
        The name of the user to be added.
    age: str
        The age of the user to be added.
    email: str
        The email of the user to be added.
    status: str
        The status of the user to be added (active/inactive).
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
    Update a user's information in the data list and save it to the file.

    This function searches for a user by name in the data list, updates
    the specified field with the new value, and saves the updated list
    to the file. If the user is not found, it logs an info message.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        The list of existing user dictionaries.
    name: str
        The name of the user to be updated.
    field: str
        The field to be updated (name/age/email/status).
    value: str
        The new value for the specified field.
    """
    user_found: bool = False
    for index, user in enumerate(data):
        if user["name"] == name:
            data[index][field] = value
            user_found = True
    if not user_found:
        logging.info("User not found")
    else:
        save_to_file(data)
        logging.info(f"Updated {name} {field}")


def get_user(
    data: list[dict[str, str | int]], name: str
) -> dict[str, str | int] | str:
    """
    Retrieve a user's information from the data list.

    This function searches for a user by name in the data list and
    returns the user's dictionary if found. If the user is not found,
    it returns "Not found".

    Parameters
    ----------
    data: list[dict[str, str | int]]
        The list of existing user dictionaries.
    name: str
        The name of the user to be retrieved.

    Returns
    -------
    dict[str, str | int] | str
        The user's dictionary if found, otherwise "Not found".
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str | int]]:
    """
    Load user data from a JSON file.

    This function checks if the "output.json" file exists and loads the
    user data from it. If the file does not exist, it returns an empty
    list.

    Returns
    -------
    list[dict[str, str | int]]
        The list of user dictionaries loaded from the file.
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def display_header() -> None:
    """
    Display the header for the user manager.

    This function uses the pyfiglet library to generate and print a
    stylized header for the user manager.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)


def get_user_input() -> str:
    """
    Display menu options and get user input.

    This function prints the available menu options and returns the
    user's choice.

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
    Handle the user's menu choice and perform the corresponding action.

    This function executes the appropriate action based on the user's
    choice and returns a boolean indicating whether to continue the
    main loop.

    Parameters
    ----------
    choice: str
        The user's menu choice.
    data: list[dict[str, str | int]]
        The list of existing user dictionaries.

    Returns
    -------
    bool
        True if the main loop should continue, False if it should exit.
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
    """Execute the main functionality of the script."""
    display_header()
    data: list[dict[str, str | int]] = load_data()
    while True:
        choice: str = get_user_input()
        if not handle_choice(choice, data):
            break


if __name__ == "__main__":
    main()
