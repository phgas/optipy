"""
User Manager Module
==================

A module for managing user data, including adding, updating, and retrieving user information.

This module provides functions to interact with a JSON file storing user data. It allows users to add new users, update existing user information, and retrieve user details. The module uses a simple command-line interface for user interaction.

Examples
--------
>>> from user_manager import add_user, save_to_file
>>> users = []
>>> add_user("John Doe", "30", "john@example.com", "active", users)
>>> save_to_file(users)
"""

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
    Save the provided user data to a JSON file.

    This function writes the given list of user dictionaries to a file named 'users.json'. It overwrites any existing data in the file.

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
    >>> users = [{"name": "John Doe", "age": "30"}]
    >>> save_to_file(users)
    """
    with open("users.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(data_to_save, indent=4))


def add_user(name: str, age: str, email: str, status: str, users: list[dict[str, str | int]]) -> None:
    """
    Add a new user to the list and save it to the file.

    This function creates a new user dictionary with the provided details, appends it to the list of users, and then saves the updated list to the file.

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
    users: list[dict[str, str | int]]
        The list of existing users to which the new user will be added.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> users = []
    >>> add_user("John Doe", "30", "john@example.com", "active", users)
    """
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status
    }
    users.append(user)
    save_to_file(users)
    logging.info(f"User added: {name}")


def update_user(name: str, field: str, value: str, users: list[dict[str, str | int]]) -> None:
    """
    Update a user's information in the list and save it to the file.

    This function searches for a user by name, updates the specified field with the new value, and then saves the updated list to the file.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.
    users: list[dict[str, str | int]]
        The list of users where the update will be applied.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> users = [{"name": "John Doe", "age": "30"}]
    >>> update_user("John Doe", "age", "31", users)
    """
    for i, user in enumerate(users):
        if user["name"] == name:
            users[i][field] = value
            save_to_file(users)
            logging.info(f"Updated {name} {field}")
            return
    logging.info("User not found")


def get_user(name: str, users: list[dict[str, str | int]]) -> dict[str, str | int] | str:
    """
    Retrieve a user's information from the list.

    This function searches for a user by name and returns the user's dictionary if found, or a string indicating the user was not found.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.
    users: list[dict[str, str | int]]
        The list of users to search through.

    Returns
    -------
    dict[str, str | int] | str
        The user's dictionary if found, or 'Not found' if the user does not exist.

    Examples
    --------
    >>> users = [{"name": "John Doe", "age": "30"}]
    >>> get_user("John Doe", users)
    {'name': 'John Doe', 'age': '30'}
    """
    for user in users:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str | int]]:
    """
    Load user data from a JSON file.

    This function reads user data from 'output.json' if the file exists, and returns an empty list if it does not.

    Returns
    -------
    list[dict[str, str | int]]
        A list of user dictionaries loaded from the file, or an empty list if the file does not exist.

    Examples
    --------
    >>> load_data()
    []
    """
    users: list[dict[str, str | int]] = []
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as f:
            users.extend(json.load(f))
    return users


def display_menu() -> str:
    """
    Display the user management menu and get user input.

    This function prints a formatted menu using pyfiglet and prompts the user for a choice.

    Returns
    -------
    str
        The user's choice as a string.

    Examples
    --------
    >>> display_menu()
    User Manager
    1. Add user
    2. Update user
    3. Get user
    4. Exit
    Enter choice: 
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")
    return input("Enter choice: ")


def process_add_user(users: list[dict[str, str | int]]) -> None:
    """
    Process the addition of a new user.

    This function prompts the user for new user details and calls the add_user function to add the user to the list.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The list of users to which the new user will be added.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> users = []
    >>> process_add_user(users)
    Enter name: John Doe
    Enter age: 30
    Enter email: john@example.com
    Enter status (active/inactive): active
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    add_user(name, age, email, status, users)


def process_update_user(users: list[dict[str, str | int]]) -> None:
    """
    Process the update of an existing user.

    This function prompts the user for the name of the user to update, the field to update, and the new value, then calls the update_user function.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The list of users where the update will be applied.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> users = [{"name": "John Doe", "age": "30"}]
    >>> process_update_user(users)
    Enter name to update: John Doe
    Enter field to update (name/age/email/status): age
    Enter new value: 31
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    update_user(name, field, value, users)


def process_get_user(users: list[dict[str, str | int]]) -> None:
    """
    Process the retrieval of a user's information.

    This function prompts the user for a name, calls the get_user function, and logs the user's details if found.

    Parameters
    ----------
    users: list[dict[str, str | int]]
        The list of users to search through.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> users = [{"name": "John Doe", "age": "30"}]
    >>> process_get_user(users)
    Enter name to find: John Doe
    """
    name: str = input("Enter name to find: ")
    result: dict[str, str | int] | str = get_user(name, users)
    if result == "Not found":
        logging.info("User not found")
    else:
        logging.info(f"Name: {result['name']}")
        logging.info(f"Age: {result['age']}")
        logging.info(f"Email: {result['email']}")
        logging.info(f"Status: {result['status']}")


def process_choice(choice: str, users: list[dict[str, str | int]]) -> bool:
    """
    Process the user's menu choice.

    This function handles the user's choice from the menu, calling the appropriate function based on the choice.

    Parameters
    ----------
    choice: str
        The user's menu choice as a string.
    users: list[dict[str, str | int]]
        The list of users to interact with.

    Returns
    -------
    bool
        True if the program should continue running, False if it should exit.

    Examples
    --------
    >>> users = []
    >>> process_choice("1", users)
    Enter name: John Doe
    Enter age: 30
    Enter email: john@example.com
    Enter status (active/inactive): active
    True
    """
    if choice == "1":
        process_add_user(users)
    elif choice == "2":
        process_update_user(users)
    elif choice == "3":
        process_get_user(users)
    elif choice == "4":
        logging.info("Exiting...")
        return False
    else:
        logging.info("Invalid choice, try again")
    return True


def main() -> None:
    """
    Execute the main functionality of the script.

    This function loads user data, displays the menu, and processes user choices in a loop until the user chooses to exit.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> main()
    """
    users: list[dict[str, str | int]] = load_data()
    while True:
        choice: str = display_menu()
        if not process_choice(choice, users):
            break


if __name__ == "__main__":
    main()
