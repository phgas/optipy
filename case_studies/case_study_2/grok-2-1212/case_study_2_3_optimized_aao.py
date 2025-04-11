"""
User Manager
============

A module for managing user data, including adding, updating, and retrieving user information.

This module provides functions to interact with a JSON file for storing and managing user data.
It includes functionality to add new users, update existing user information, and retrieve user details.

Examples
--------
>>> from user_manager import add_user, update_user, get_user
>>> add_user("John Doe", 30, "john@example.com", "active")
User added: John Doe

>>> update_user("John Doe", "email", "john.doe@example.com")
Updated John Doe email

>>> user = get_user("John Doe")
>>> print(user["name"], user["email"])
John Doe john.doe@example.com
"""


import json
import os
import pyfiglet

# Standard library imports
import json
import os

# Third party imports
import pyfiglet

data: list[dict[str, str | int]] = []


def save_to_file(data_to_save: list[dict[str, str | int]]) -> None:
    """
    Save the given data to a JSON file.

    This function writes the provided data to 'users.json' in a formatted manner.

    Parameters
    ----------
    data_to_save: list[dict[str, str | int]]
        The list of user dictionaries to be saved.

    Returns
    -------
    None
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(name: str, age: int, email: str, status: str) -> None:
    """
    Add a new user to the data list and save it to the file.

    This function creates a new user dictionary with the provided details, adds it to the data list,
    and then saves the updated list to the file.

    Parameters
    ----------
    name: str
        The name of the user.
    age: int
        The age of the user.
    email: str
        The email address of the user.
    status: str
        The status of the user (active/inactive).

    Returns
    -------
    None
    """
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status
    }
    data.append(user)
    save_to_file(data)
    print(f"User added: {name}")


def update_user(name: str, field: str, value: str | int) -> None:
    """
    Update a user's information based on their name.

    This function searches for a user by name, updates the specified field with the given value,
    and saves the updated data to the file.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str | int
        The new value for the specified field.

    Returns
    -------
    None
    """
    user_found: bool = False
    for user in data:
        if user["name"] == name:
            user[field] = value
            user_found = True
    if not user_found:
        print("User not found")
    else:
        save_to_file(data)
        print(f"Updated {name} {field}")


def get_user(name: str) -> dict[str, str | int] | str:
    """
    Retrieve a user's information based on their name.

    This function searches for a user by name and returns their details if found.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict[str, str | int] | str
        The user's details as a dictionary if found, otherwise "Not found".
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> None:
    """
    Load user data from the JSON file if it exists.

    This function checks for the existence of 'users.json' and loads its contents into the data list.

    Returns
    -------
    None
    """
    if os.path.exists("users.json"):
        with open("users.json", "r", encoding="utf-8") as file:
            data.clear()
            data.extend(json.load(file))


def main() -> None:
    """
    Execute the main functionality of the User Manager.

    This function runs the main program loop, allowing users to interact with the user management system.

    Returns
    -------
    None
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    while True:
        print("1. Add user")
        print("2. Update user")
        print("3. Get user")
        print("4. Exit")
        choice: str = input("Enter choice: ")

        if choice == "1":
            name: str = input("Enter name: ")
            age: int = int(input("Enter age: "))
            email: str = input("Enter email: ")
            status: str = input("Enter status (active/inactive): ")
            add_user(name, age, email, status)
        elif choice == "2":
            name: str = input("Enter name to update: ")
            field: str = input(
                "Enter field to update (name/age/email/status): ")
            value: str | int = input("Enter new value: ")
            if field == "age":
                value = int(value)
            update_user(name, field, value)
        elif choice == "3":
            name: str = input("Enter name to find: ")
            result: dict[str, str | int] | str = get_user(name)
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

        load_data()


if __name__ == "__main__":
    main()
