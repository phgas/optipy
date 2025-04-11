"""
User Management System
=====================

A module for managing user data, including adding, updating, and retrieving user information.

This module provides functions to interact with a JSON file storing user data. It allows users to be added, updated, and searched within a simple command-line interface.

Examples
--------
>>> from user_management import add_user, get_user
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
    format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


def save_to_file(data_to_save: list[dict[str, str | int]]) -> None:
    """
    Save the provided data to a JSON file.

    This function writes the given data to 'users.json' in a formatted manner.

    Parameters
    ----------
    data_to_save: list[dict[str, str | int]]
        A list of dictionaries containing user data to be saved.

    Returns
    -------
    None
        This function does not return anything; it writes data to a file.

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
        f.write(json.dumps(data_to_save, indent=4))


def add_user(name: str, age: str, email: str, status: str) -> None:
    """
    Add a new user to the data file.

    This function creates a new user entry and appends it to the existing data, then saves the updated data.

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
        This function does not return anything; it modifies the data file.

    Examples
    --------
    >>> add_user("Jane Doe", "25", "jane@example.com", "active")
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

    This function searches for a user by name and updates the specified field with the new value.

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
        This function does not return anything; it modifies the data file.

    Examples
    --------
    >>> update_user("John Doe", "email", "john.new@example.com")
    """
    data: list[dict[str, str | int]] = load_data()
    found: bool = False
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            found = True
    if not found:
        logging.info("User not found")
    else:
        save_to_file(data)
        logging.info(f"Updated {name} {field}")


def get_user(name: str) -> dict[str, str | int] | str:
    """
    Retrieve user information from the data file.

    This function searches for a user by name and returns their information if found.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.

    Returns
    -------
    dict[str, str | int] | str
        A dictionary containing the user's information if found, or 'Not found' if the user does not exist.

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

    This function reads the 'users.json' file and returns its contents as a list of dictionaries.

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
    Display the menu options and get user's choice.

    This function prints the menu options and returns the user's input.

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

    This function prompts the user for name, age, email, and status, and returns them as a tuple.

    Returns
    -------
    tuple[str, str, str, str]
        A tuple containing the user's input for name, age, email, and status.

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

    This function prompts the user for the name of the user to update, the field to update, and the new value.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing the user's input for name, field, and value.

    Examples
    --------
    >>> name, field, value = get_update_input()
    >>> print(name, field, value)
    John Doe email john.new@example.com
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return name, field, value


def get_search_input() -> str:
    """
    Get user input for searching a user.

    This function prompts the user for the name of the user to search.

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


def display_user_info(user: dict[str, str | int]) -> None:
    """
    Display user information.

    This function logs the user's information including name, age, email, and status.

    Parameters
    ----------
    user: dict[str, str | int]
        A dictionary containing the user's information.

    Returns
    -------
    None
        This function does not return anything; it logs information.

    Examples
    --------
    >>> user = {'name': 'John Doe', 'age': '30', 'email': 'john@example.com', 'status': 'active'}
    >>> display_user_info(user)
    """
    logging.info(f"Name: {user['name']}")
    logging.info(f"Age: {user['age']}")
    logging.info(f"Email: {user['email']}")
    logging.info(f"Status: {user['status']}")


def main() -> None:
    """
    Execute the main functionality of the script.

    This function runs the user management system, displaying a menu and processing user choices.

    Returns
    -------
    None
        This function does not return anything; it runs the main loop of the program.

    Examples
    --------
    >>> main()
    """
    import pyfiglet

    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    while True:
        choice: str = display_menu()

        if choice == "1":
            name: str
            age: str
            email: str
            status: str
            name, age, email, status = get_user_input()
            add_user(name, age, email, status)
        elif choice == "2":
            name: str
            field: str
            value: str
            name, field, value = get_update_input()
            update_user(name, field, value)
        elif choice == "3":
            name: str = get_search_input()
            result: dict[str, str | int] | str = get_user(name)
            if result == "Not found":
                logging.info("User not found")
            else:
                display_user_info(result)
        elif choice == "4":
            logging.info("Exiting...")
            break
        else:
            logging.info("Invalid choice, try again")


if __name__ == "__main__":
    main()
