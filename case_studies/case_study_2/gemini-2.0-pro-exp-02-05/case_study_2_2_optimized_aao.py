"""
User manager
============

A simple module for managing users.

This module provides functionalities to add, update, and retrieve user
information. It uses a JSON file to store user data and provides a command-line
interface for user interaction.

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
import os

# Third party imports
import pyfiglet  # type: ignore


def save_to_file(data_to_save: list[dict[str, str]]) -> None:
    """
    Save the given data to a JSON file.

    Parameters
    ----------
    data_to_save : list[dict[str, str]]
        The data to be saved.

    Returns
    -------
    None
        This function does not return a value.
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
    name : str
        The name of the user.
    age : str
        The age of the user.
    email : str
        The email of the user.
    status : str
        The status of the user (active/inactive).
    data : list[dict[str, str]]
        The list of current users.

    Returns
    -------
    None
        This function does not return a value.
    """
    user: dict[str, str] = {}
    user["name"] = name
    user["age"] = age
    user["email"] = email
    user["status"] = status
    data.append(user)
    save_to_file(data)
    print(f"User added: {name}")


def update_user(
    name: str, field: str, value: str, data: list[dict[str, str]]
) -> None:
    """
    Update user information based on the provided name, field, and value.

    Parameters
    ----------
    name : str
        The name of the user to update.
    field : str
        The field to update (name/age/email/status).
    value : str
        The new value for the field.
    data : list[dict[str, str]]
        The list of current users.

    Returns
    -------
    None
        This function does not return a value.
    """
    user_was_found: bool = False
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            user_was_found = True
    if not user_was_found:
        print("User not found")
    else:
        save_to_file(data)
        print(f"Updated {name} {field}")


def get_user(name: str, data: list[dict[str, str]]) -> dict[str, str] | str:
    """
    Find and return a user by name.

    Parameters
    ----------
    name : str
        The name of the user to find.
    data : list[dict[str, str]]
        The list of current users.

    Returns
    -------
    dict[str, str] | str
        The user dictionary if found, "Not found" otherwise.
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def get_choice() -> str:
    """
    Get and return the user's choice from the menu.

    Returns
    -------
    str
        The user's choice.
    """
    return input("Enter choice: ")


def get_user_input() -> tuple[str, str, str, str]:
    """
    Get and return user details from input.

    Returns
    -------
    tuple[str, str, str, str]
        A tuple containing name, age, email, and status.
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return name, age, email, status


def get_update_input() -> tuple[str, str, str]:
    """
    Get and return update details from input.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing name, field, and value.
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return name, field, value


def print_user_details(user: dict[str, str]) -> None:
    """
    Print the details of a user.

    Parameters
    ----------
    user : dict[str, str]
        The user dictionary.

    Returns
    -------
    None
        This function does not return a value.
    """
    print(f"Name: {user['name']}")
    print(f"Age: {user['age']}")
    print(f"Email: {user['email']}")
    print(f"Status: {user['status']}")


def load_data() -> list[dict[str, str]]:
    """
    Load and return user data from the JSON file.

    Returns
    -------
    list[dict[str, str]]
        The loaded user data.
    """
    data: list[dict[str, str]] = []
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            data.clear()
            data.extend(json.load(file))
    return data


def main() -> None:
    """Executes the main functionality of the script."""
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    data: list[dict[str, str]] = load_data()
    while True:
        print("1. Add user")
        print("2. Update user")
        print("3. Get user")
        print("4. Exit")
        choice: str = get_choice()

        if choice == "1":
            name, age, email, status = get_user_input()
            add_user(name, age, email, status, data)
        elif choice == "2":
            name, field, value = get_update_input()
            update_user(name, field, value, data)
        elif choice == "3":
            name: str = input("Enter name to find: ")
            result: dict[str, str] | str = get_user(name, data)
            if result == "Not found":
                print("User not found")
            else:
                print_user_details(result)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice, try again")


if __name__ == "__main__":
    main()
