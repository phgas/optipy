"""
User Manager
============

A simple module for managing user data.

This module provides functionalities to add, update, and retrieve user
information. It uses a JSON file to store user data and supports basic
operations like adding a new user, updating existing user information,
and retrieving user details by their name.

Examples
--------
>>> from user_manager import add_user, update_user, get_user
>>> users = []
>>> users = add_user("John Doe", "30", "john.doe@example.com", "active", users)
>>> print(users)
[{'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com', \
'status': 'active'}]

>>> users = update_user("John Doe", "age", "31", users)
>>> print(users)
[{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com', \
'status': 'active'}]

>>> user = get_user("John Doe", users)
>>> print(user)
{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com', \
'status': 'active'}
"""

import json
import logging
import os

import pyfiglet


logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def save_to_file(data_to_save: list[dict]) -> None:
    """
    Save the given data to a JSON file named 'users.json'.

    This function serializes the provided data and writes it to a JSON
    file. It uses an indentation of 4 spaces for better readability.

    Parameters
    ----------
    data_to_save: list[dict]
        The data to be saved to the file.

    Returns
    -------
    None

    Examples
    --------
    >>> save_to_file([{'name': 'John Doe', 'age': '30'}])
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(
    name: str, age: str, email: str, status: str, users: list[dict]
) -> list[dict]:
    """
    Add a new user to the list of users and save it to the file.

    This function creates a new user dictionary with the provided
    information, appends it to the list of users, and then saves the
    updated list to a JSON file.

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
    users: list[dict]
        The current list of users.

    Returns
    -------
    list[dict]
        The updated list of users with the new user added.

    Examples
    --------
    >>> users = []
    >>> add_user('Jane Smith', '25', 'jane.smith@example.com', 'active', users)
    [{'name': 'Jane Smith', 'age': '25', 'email': \
'jane.smith@example.com', 'status': 'active'}]
    """
    user: dict = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    users.append(user)
    save_to_file(users)
    logging.info(f"User added: {name}")
    return users


def update_user(
    name: str, field: str, value: str, users: list[dict]
) -> list[dict]:
    """
    Update a user's information based on their name and save it to the
    file.

    This function searches for a user by their name and updates the
    specified field with the new value. If the user is found, the
    updated list is saved to a JSON file.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.
    users: list[dict]
        The current list of users.

    Returns
    -------
    list[dict]
        The updated list of users, or the original list if the user was
        not found.

    Examples
    --------
    >>> users = [{'name': 'John Doe', 'age': '30', 'email': \
'john.doe@example.com', 'status': 'active'}]
    >>> update_user('John Doe', 'age', '31', users)
    [{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com', \
'status': 'active'}]
    """
    for i, user in enumerate(users):
        if user["name"] == name:
            users[i][field] = value
            save_to_file(users)
            logging.info(f"Updated {name} {field}")
            return users
    logging.info("User not found")
    return users


def get_user(name: str, users: list[dict]) -> dict | str:
    """
    Retrieve a user's information based on their name.

    This function searches for a user by their name and returns their
    information. If the user is not found, it returns the string 'Not
    found'.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.
    users: list[dict]
        The current list of users.

    Returns
    -------
    dict | str
        The user's information as a dictionary if found, otherwise the
        string 'Not found'.

    Examples
    --------
    >>> users = [{'name': 'John Doe', 'age': '30', 'email': \
'john.doe@example.com', 'status': 'active'}]
    >>> get_user('John Doe', users)
    {'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com', \
'status': 'active'}

    >>> get_user('Jane Smith', users)
    'Not found'
    """
    for user in users:
        if user["name"] == name:
            return user
    return "Not found"


def handle_add_user(users: list[dict]) -> list[dict]:
    """
    Handle the process of adding a new user.

    This function prompts the user for the new user's information,
    then calls the add_user function to add the user to the list.

    Parameters
    ----------
    users: list[dict]
        The current list of users.

    Returns
    -------
    list[dict]
        The updated list of users with the new user added.

    Examples
    --------
    (Assuming user inputs 'Jane Smith', '25', 'jane.smith@example.com',
    'active')
    >>> users = []
    >>> handle_add_user(users)
    [{'name': 'Jane Smith', 'age': '25', 'email': \
'jane.smith@example.com', 'status': 'active'}]
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return add_user(name, age, email, status, users)


def handle_update_user(users: list[dict]) -> list[dict]:
    """
    Handle the process of updating an existing user.

    This function prompts the user for the name of the user to update,
    the field to update, and the new value, then calls the update_user
    function to update the user's information.

    Parameters
    ----------
    users: list[dict]
        The current list of users.

    Returns
    -------
    list[dict]
        The updated list of users, or the original list if the user was
        not found.

    Examples
    --------
    (Assuming user inputs 'John Doe', 'age', '31')
    >>> users = [{'name': 'John Doe', 'age': '30', 'email': \
'john.doe@example.com', 'status': 'active'}]
    >>> handle_update_user(users)
    [{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com', \
'status': 'active'}]
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return update_user(name, field, value, users)


def handle_get_user(users: list[dict]) -> None:
    """
    Handle the process of retrieving and displaying a user's
    information.

    This function prompts the user for the name of the user to retrieve,
    then calls the get_user function to retrieve the user's information
    and displays it.

    Parameters
    ----------
    users: list[dict]
        The current list of users.

    Returns
    -------
    None

    Examples
    --------
    (Assuming user inputs 'John Doe')
    >>> users = [{'name': 'John Doe', 'age': '30', 'email': \
'john.doe@example.com', 'status': 'active'}]
    >>> handle_get_user(users)
    Name: John Doe
    Age: 30
    Email: john.doe@example.com
    Status: active

    (Assuming user inputs 'Jane Smith')
    >>> handle_get_user(users)
    User not found
    """
    name: str = input("Enter name to find: ")
    result: dict | str = get_user(name, users)
    if result == "Not found":
        logging.info("User not found")
    else:
        logging.info(f"Name: {result['name']}")
        logging.info(f"Age: {result['age']}")
        logging.info(f"Email: {result['email']}")
        logging.info(f"Status: {result['status']}")


def load_data_from_file() -> list:
    """
    Load user data from 'output.json' if it exists.

    This function checks if the 'output.json' file exists and, if so,
    loads the user data from it. If the file does not exist, it returns
    an empty list.

    Parameters
    ----------
    None

    Returns
    -------
    list
        The list of users loaded from the file, or an empty list if the
        file does not exist.

    Examples
    --------
    (Assuming 'output.json' contains [{'name': 'John Doe', 'age': '30'}])
    >>> load_data_from_file()
    [{'name': 'John Doe', 'age': '30'}]

    (Assuming 'output.json' does not exist)
    >>> load_data_from_file()
    []
    """
    users: list = []
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            users = json.load(file)
    return users


def display_menu() -> None:
    """
    Display the main menu options to the user.

    This function prints the available options to the console.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> display_menu()
    1. Add user
    2. Update user
    3. Get user
    4. Exit
    """
    print("1. Add user")
    print("2. Update user")
    print("3. Get user")
    print("4. Exit")


def process_choice(choice: str, users: list[dict]) -> list[dict] | None:
    """
    Process the user's choice and call the appropriate handler.

    This function takes the user's choice and calls the corresponding
    handler function. If the choice is invalid, it logs a message.

    Parameters
    ----------
    choice: str
        The user's choice (e.g., '1', '2', '3', '4').
    users: list[dict]
        The current list of users.

    Returns
    -------
    list[dict] | None
        The updated list of users if the choice was '1' or '2',
        None if the choice was '4' (to exit),
        or the original list if the choice was invalid.

    Examples
    --------
    (Assuming user inputs '1')
    >>> users = []
    >>> process_choice('1', users)
    (Calls handle_add_user and returns the updated list)

    (Assuming user inputs '4')
    >>> process_choice('4', users)
    (Logs "Exiting..." and returns None)

    (Assuming user inputs '5')
    >>> process_choice('5', users)
    (Logs "Invalid choice, try again" and returns the original list)
    """
    if choice == "1":
        return handle_add_user(users)
    elif choice == "2":
        return handle_update_user(users)
    elif choice == "3":
        handle_get_user(users)
    elif choice == "4":
        logging.info("Exiting...")
        return None
    else:
        logging.info("Invalid choice, try again")
    return users


def main() -> None:
    """
    Execute the main functionality of the script.

    This function displays a header, loads user data from a file,
    and enters a loop to display the menu, process user input,
    and perform actions based on the user's choices.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    (Assuming user interacts with the menu and provides input)
    >>> main()
    (Displays header, menu, and processes user choices)
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    users: list = load_data_from_file()
    while True:
        display_menu()
        choice: str = input("Enter choice: ")
        users = process_choice(choice, users)
        if users is None:
            break


if __name__ == "__main__":
    main()
