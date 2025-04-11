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
import os

# Third party imports
import pyfiglet  # type: ignore


def save_to_file(data: list[dict[str, str | int]]) -> None:
    """
    Save the provided user data to a JSON file.

    This function writes the given list of user dictionaries to a file
    named "users.json" in JSON format with an indentation of 4 spaces.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of dictionaries containing user information.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4))


def add_user(
    data: list[dict[str, str | int]], name: str, age: str, email: str, status: str
) -> None:
    """
    Add a new user to the data list and save it to the file.

    This function creates a new user dictionary with the provided
    information, appends it to the data list, and saves the updated
    list to the file.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of dictionaries containing user information.
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
    data.append(user)
    save_to_file(data)
    print(f"User added: {name}")


def update_user(
    data: list[dict[str, str | int]], name: str, field: str, value: str
) -> None:
    """
    Update a specific field of a user in the data list and save it to
    the file.

    This function searches for a user by name in the data list, updates
    the specified field with the new value, and saves the updated list
    to the file. If the user is not found, it prints a message
    indicating so.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of dictionaries containing user information.
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str
        The new value for the specified field.
    """
    user_found: bool = False
    for index, user in enumerate(data):
        if user["name"] == name:
            data[index][field] = value
            user_found = True
    if not user_found:
        print("User not found")
    else:
        save_to_file(data)
        print(f"Updated {name} {field}")


def get_user(
    data: list[dict[str, str | int]], name: str
) -> dict[str, str | int] | str:
    """
    Retrieve a user by name from the data list.

    This function searches for a user by name in the data list and
    returns the user dictionary if found. If the user is not found, it
    returns "Not found".

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of dictionaries containing user information.
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict[str, str | int] | str
        The user dictionary if found, otherwise "Not found".
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str | int]]:
    """
    Load user data from a JSON file.

    This function checks if a file named "output.json" exists. If it
    does, it reads and returns the user data from the file. If the file
    does not exist, it returns an empty list.

    Returns
    -------
    list[dict[str, str | int]]
        A list of dictionaries containing user information.
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def display_header() -> None:
    """
    Display the header for the user management system.

    This function uses the pyfiglet library to generate and print a
    stylized header for the user management system.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)


def get_user_choice() -> str:
    """
    Display the menu options and get the user's choice.

    This function prints the available menu options and prompts the user
    to enter their choice.

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


def handle_user_choice(
    data: list[dict[str, str | int]], user_choice: str
) -> bool:
    """
    Handle the user's choice and perform the corresponding action.

    This function takes the user's choice and performs the corresponding
    action, such as adding, updating, or retrieving a user. It returns
    False if the user chooses to exit, otherwise it returns True.

    Parameters
    ----------
    data: list[dict[str, str | int]]
        A list of dictionaries containing user information.
    user_choice: str
        The user's choice as a string.

    Returns
    -------
    bool
        False if the user chooses to exit, otherwise True.
    """
    if user_choice == "1":
        name: str = input("Enter name: ")
        age: str = input("Enter age: ")
        email: str = input("Enter email: ")
        status: str = input("Enter status (active/inactive): ")
        add_user(data, name, age, email, status)
    elif user_choice == "2":
        name: str = input("Enter name to update: ")
        field: str = input("Enter field to update (name/age/email/status): ")
        value: str = input("Enter new value: ")
        update_user(data, name, field, value)
    elif user_choice == "3":
        name: str = input("Enter name to find: ")
        result = get_user(data, name)
        if result == "Not found":
            print("User not found")
        else:
            print(f"Name: {result['name']}")
            print(f"Age: {result['age']}")
            print(f"Email: {result['email']}")
            print(f"Status: {result['status']}")
    elif user_choice == "4":
        print("Exiting...")
        return False
    else:
        print("Invalid choice, try again")
    return True


def main() -> None:
    """Executes the main functionality of the script."""
    display_header()
    data: list[dict[str, str | int]] = load_data()
    while True:
        user_choice: str = get_user_choice()
        if not handle_user_choice(data, user_choice):
            break


if __name__ == "__main__":
    main()
