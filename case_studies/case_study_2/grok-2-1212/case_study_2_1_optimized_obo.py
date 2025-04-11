"""
User Manager Module
==================

A module for managing user data, including adding, updating, and retrieving user information.

This module provides functions to interact with a JSON file storing user data. It allows users to add new users, update existing user information, and retrieve user details. The module uses a simple command-line interface for user interaction.

Examples
--------
>>> from user_manager import add_user, get_user
>>> add_user("John Doe", "30", "john@example.com", "active")
>>> user = get_user("John Doe")
>>> print(user)
{'name': 'John Doe', 'age': '30', 'email': 'john@example.com', 'status': 'active'}
"""


# Standard library imports
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def save_to_file(data_to_save: list[dict[str, str | int]]) -> None:
    """
    Save the provided data to a JSON file.

    This function writes the given data to a file named 'users.json'. It overwrites any existing data in the file.

    Parameters
    ----------
    data_to_save: list[dict[str, str | int]]
        A list of dictionaries containing user data to be saved.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    IOError
        If there is an error writing to the file.

    Examples
    --------
    >>> data = [{"name": "John", "age": "30"}]
    >>> save_to_file(data)
    """
    with open("users.json", "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)


def add_user(name: str, age: str, email: str, status: str) -> None:
    """
    Add a new user to the data file.

    This function creates a new user dictionary with the provided details, adds it to the existing data, and saves the updated data to the file.

    Parameters
    ----------
    name: str
        The name of the user.
    age: str
        The age of the user.
    email: str
        The email address of the user.
    status: str
        The status of the user (e.g., 'active' or 'inactive').

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> add_user("John Doe", "30", "john@example.com", "active")
    """
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status
    }
    data: list[dict[str, str | int]] = load_data()
    data.append(user)
    save_to_file(data)
    logging.info(f"User added: {name}")


def update_user(name: str, field: str, value: str) -> None:
    """
    Update a user's information in the data file.

    This function searches for a user by name, updates the specified field with the new value, and saves the updated data to the file.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> update_user("John Doe", "email", "newemail@example.com")
    """
    data: list[dict[str, str | int]] = load_data()
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            save_to_file(data)
            logging.info(f"Updated {name} {field}")
            return
    logging.info("User not found")


def get_user(name: str) -> dict[str, str | int] | str:
    """
    Retrieve a user's information from the data file.

    This function searches for a user by name and returns their details if found, or a 'Not found' message if the user does not exist.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict[str, str | int] | str
        A dictionary containing the user's details if found, or 'Not found' if the user does not exist.

    Examples
    --------
    >>> user = get_user("John Doe")
    >>> print(user)
    {'name': 'John Doe', 'age': '30', 'email': 'john@example.com', 'status': 'active'}
    """
    data: list[dict[str, str | int]] = load_data()
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str | int]]:
    """
    Load user data from the JSON file.

    This function reads the 'users.json' file and returns the data as a list of dictionaries. If the file does not exist, it returns an empty list.

    Returns
    -------
    list[dict[str, str | int]]
        A list of dictionaries containing user data.

    Raises
    ------
    json.JSONDecodeError
        If the JSON file is not properly formatted.

    Examples
    --------
    >>> data = load_data()
    >>> print(data)
    [{'name': 'John Doe', 'age': '30', 'email': 'john@example.com', 'status': 'active'}]
    """
    if os.path.exists("users.json"):
        with open("users.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def display_menu() -> str:
    """
    Display the menu options and get user input.

    This function prints the available options and prompts the user to enter their choice.

    Returns
    -------
    str
        The user's choice as a string.

    Examples
    --------
    >>> choice = display_menu()
    >>> print(choice)
    1
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")
    return input("Enter choice: ")


def get_user_input() -> tuple[str, str, str, str]:
    """
    Get user input for adding a new user.

    This function prompts the user to enter details for a new user and returns them as a tuple.

    Returns
    -------
    tuple[str, str, str, str]
        A tuple containing the user's name, age, email, and status.

    Examples
    --------
    >>> name, age, email, status = get_user_input()
    >>> print(name, age, email, status)
    John Doe 30 john@example.com active
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return name, age, email, status


def get_update_input() -> tuple[str, str, str]:
    """
    Get user input for updating a user's information.

    This function prompts the user to enter the name of the user to update, the field to update, and the new value.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing the user's name, the field to update, and the new value.

    Examples
    --------
    >>> name, field, value = get_update_input()
    >>> print(name, field, value)
    John Doe email newemail@example.com
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return name, field, value


def get_search_input() -> str:
    """
    Get user input for searching a user.

    This function prompts the user to enter the name of the user to search for.

    Returns
    -------
    str
        The name of the user to search for.

    Examples
    --------
    >>> name = get_search_input()
    >>> print(name)
    John Doe
    """
    return input("Enter name to find: ")


def process_choice(choice: str) -> bool:
    """
    Process the user's menu choice and perform the corresponding action.

    This function handles the logic for each menu option, calling the appropriate functions based on the user's choice.

    Parameters
    ----------
    choice: str
        The user's menu choice as a string.

    Returns
    -------
    bool
        True if the program should continue running, False if it should exit.

    Examples
    --------
    >>> continue_running = process_choice("1")
    >>> print(continue_running)
    True
    """
    if choice == "1":
        name, age, email, status = get_user_input()
        add_user(name, age, email, status)
    elif choice == "2":
        name, field, value = get_update_input()
        update_user(name, field, value)
    elif choice == "3":
        name: str = get_search_input()
        result: dict[str, str | int] | str = get_user(name)
        if result == "Not found":
            logging.info("User not found")
        else:
            logging.info(f"Name: {result['name']}")
            logging.info(f"Age: {result['age']}")
            logging.info(f"Email: {result['email']}")
            logging.info(f"Status: {result['status']}")
    elif choice == "4":
        logging.info("Exiting...")
        return False
    else:
        logging.info("Invalid choice, try again")
    return True


def main() -> None:
    """
    Execute the main functionality of the script.

    This function runs the user management interface, displaying the menu and processing user choices until the user chooses to exit.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> main()
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    while True:
        choice: str = display_menu()
        if not process_choice(choice):
            break


if __name__ == "__main__":
    main()
