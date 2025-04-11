"""
User Management Module
======================

A module for managing user data, including adding, updating, retrieving,
and saving user information.

This module provides functions to manage user data stored in a JSON file.
It allows adding new users, updating existing user information, retrieving
user details, and displaying a menu for user interaction.

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


def save_to_file(data: list[dict[str, str]]) -> None:
    """
    Save the provided user data to a JSON file.

    This function writes the given list of user dictionaries to a file named
    "users.json" in a pretty-printed JSON format.

    Parameters
    ----------
    data: list[dict[str, str]]
        A list of user dictionaries to be saved.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4))


def add_user(data: list[dict[str, str]], name: str, age: str, email: str,
             status: str) -> None:
    """
    Add a new user to the data list and save it to the file.

    This function creates a new user dictionary with the provided details,
    appends it to the existing data list, and saves the updated list to the
    file.

    Parameters
    ----------
    data: list[dict[str, str]]
        The existing list of user dictionaries.
    name: str
        The name of the user to be added.
    age: str
        The age of the user to be added.
    email: str
        The email of the user to be added.
    status: str
        The status of the user to be added (e.g., active/inactive).
    """
    user: dict[str, str] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status
    }
    data.append(user)
    save_to_file(data)
    print(f"User added: {name}")


def update_user(data: list[dict[str, str]], name: str, field: str,
                value: str) -> None:
    """
    Update a specific field of a user in the data list and save it to the file.

    This function searches for a user by name in the data list, updates the
    specified field with the new value, and saves the updated list to the file.
    If the user is not found, it prints a message indicating so.

    Parameters
    ----------
    data: list[dict[str, str]]
        The existing list of user dictionaries.
    name: str
        The name of the user to be updated.
    field: str
        The field to be updated (e.g., name, age, email, status).
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


def get_user(data: list[dict[str, str]], name: str) -> dict[str, str] | str:
    """
    Retrieve a user's details from the data list by name.

    This function searches for a user by name in the data list and returns the
    user's dictionary if found. If the user is not found, it returns a
    "Not found" message.

    Parameters
    ----------
    data: list[dict[str, str]]
        The existing list of user dictionaries.
    name: str
        The name of the user to be retrieved.

    Returns
    -------
    dict[str, str] | str
        The user's dictionary if found, otherwise "Not found".
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str]]:
    """
    Load user data from a JSON file.

    This function checks if the "output.json" file exists and loads the user
    data from it. If the file does not exist, it returns an empty list.

    Returns
    -------
    list[dict[str, str]]
        The list of user dictionaries loaded from the file.
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def display_header() -> None:
    """
    Display the header for the user management application.

    This function uses the pyfiglet library to generate and print a stylized
    header for the user management application.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)


def display_menu() -> None:
    """
    Display the menu options for the user management application.

    This function prints the available menu options for the user to interact
    with the user management application.
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")


def main() -> None:
    """Executes the main functionality of the script."""
    display_header()
    data: list[dict[str, str]] = load_data()
    while True:
        display_menu()
        choice: str = input("Enter choice: ")

        if choice == "1":
            name: str = input("Enter name: ")
            age: str = input("Enter age: ")
            email: str = input("Enter email: ")
            status: str = input("Enter status (active/inactive): ")
            add_user(data, name, age, email, status)
        elif choice == "2":
            name: str = input("Enter name to update: ")
            field: str = input(
                "Enter field to update (name/age/email/status): ")
            value: str = input("Enter new value: ")
            update_user(data, name, field, value)
        elif choice == "3":
            name: str = input("Enter name to find: ")
            result = get_user(data, name)
            if result == "Not found":
                print("User not found")
            else:
                print(f"Name: {result['name']}")
                print(f"Age: {result['age']}")
                print(f"Email: {result['email']}")
                print(f"Status: {result['status']}")
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice, try again")


if __name__ == "__main__":
    main()
