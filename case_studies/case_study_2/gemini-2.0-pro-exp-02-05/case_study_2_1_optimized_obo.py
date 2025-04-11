"""
User Manager
============

A simple module for managing user data.

This module provides functionalities to add, update, and retrieve user
information. It uses a JSON file to store user data and supports basic
operations like adding a new user, updating existing user information,
and retrieving user details by name.

Examples
--------
>>> from user_manager import add_user, update_user, get_user
>>> data = []
>>> data = add_user("John Doe", "30", "john.doe@example.com", "active", data)
>>> print(data)
[{'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com',
'status': 'active'}]

>>> data = update_user("John Doe", "age", "31", data)
>>> print(data)
[{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com',
'status': 'active'}]

>>> user = get_user("John Doe", data)
>>> print(user)
{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com',
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


def save_to_file(data_to_save: list) -> None:
    """
    Save the provided data to a JSON file named 'users.json'.

    This function serializes the given data and writes it to 'users.json'
    with an indentation of 4 spaces.

    Parameters
    ----------
    data_to_save: list
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


def add_user(name: str, age: str, email: str, status: str, data: list) -> list:
    """
    Add a new user to the data list and save it to a file.

    This function creates a new user dictionary with the given parameters,
    appends it to the data list, saves the updated list to a file,
    and logs an info message.

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
    data: list
        The current list of user data.

    Returns
    -------
    list
        The updated list of user data with the new user added.

    Examples
    --------
    >>> data = []
    >>> add_user('Jane Smith', '25', 'jane.smith@example.com', 'active', data)
    [{'name': 'Jane Smith', 'age': '25', 'email': 'jane.smith@example.com',
    'status': 'active'}]
    """
    user: dict = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    data.append(user)
    save_to_file(data)
    logging.info(f"User added: {name}")
    return data


def update_user(name: str, field: str, value: str, data: list) -> list:
    """
    Update the specified field of a user in the data list and save it to a
    file.

    This function searches for a user by name, updates the specified field
    with the new value, saves the updated list to a file, and logs an info
    message. If the user is not found, it logs an info message indicating
    that the user was not found.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.
    data: list
        The current list of user data.

    Returns
    -------
    list
        The updated list of user data.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> update_user('John Doe', 'age', '31', data)
    [{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com',
    'status': 'active'}]
    """
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            save_to_file(data)
            logging.info(f"Updated {name} {field}")
            return data
    logging.info("User not found")
    return data


def get_user(name: str, data: list) -> dict | str:
    """
    Retrieve a user from the data list by their name.

    This function searches for a user by name and returns the user
    dictionary if found. If the user is not found, it returns the string
    'Not found'.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.
    data: list
        The current list of user data.

    Returns
    -------
    dict | str
        The user dictionary if found, otherwise the string 'Not found'.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> get_user('John Doe', data)
    {'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com',
    'status': 'active'}

    >>> get_user('Jane Smith', data)
    'Not found'
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def handle_add_user(data: list) -> list:
    """
    Handle the process of adding a new user by taking inputs and calling the
    add_user function.

    This function prompts the user for the new user's details (name, age,
    email, and status), then calls the add_user function to add the new user
    to the data list.

    Parameters
    ----------
    data: list
        The current list of user data.

    Returns
    -------
    list
        The updated list of user data with the new user added.

    Examples
    --------
    >>> data = []
    >>> handle_add_user(data)  # User inputs: 'Jane Smith', '25',
    ... # 'jane.smith@example.com', 'active'
    [{'name': 'Jane Smith', 'age': '25', 'email': 'jane.smith@example.com',
    'status': 'active'}]
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return add_user(name, age, email, status, data)


def handle_update_user(data: list) -> list:
    """
    Handle the process of updating a user by taking inputs and calling the
    update_user function.

    This function prompts the user for the name of the user to update, the
    field to update, and the new value, then calls the update_user function
    to update the user in the data list.

    Parameters
    ----------
    data: list
        The current list of user data.

    Returns
    -------
    list
        The updated list of user data.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> handle_update_user(data)  # User inputs: 'John Doe', 'age', '31'
    [{'name': 'John Doe', 'age': '31', 'email': 'john.doe@example.com',
    'status': 'active'}]
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return update_user(name, field, value, data)


def handle_get_user(data: list) -> None:
    """
    Handle the process of retrieving and displaying a user by taking inputs
    and calling the get_user function.

    This function prompts the user for the name of the user to find, then
    calls the get_user function to retrieve the user from the data list.
    If the user is found, it displays the user's details; otherwise, it logs
    an info message indicating that the user was not found.

    Parameters
    ----------
    data: list
        The current list of user data.

    Returns
    -------
    None

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> handle_get_user(data)  # User inputs 'John Doe'
    Name: John Doe
    Age: 30
    Email: john.doe@example.com
    Status: active

    >>> handle_get_user(data)  # User inputs 'Jane Smith'
    User not found
    """
    name: str = input("Enter name to find: ")
    result: dict | str = get_user(name, data)
    if result == "Not found":
        logging.info("User not found")
    else:
        logging.info(f"Name: {result['name']}")
        logging.info(f"Age: {result['age']}")
        logging.info(f"Email: {result['email']}")
        logging.info(f"Status: {result['status']}")


def load_data_from_file() -> list:
    """
    Load user data from a JSON file named 'output.json'.

    This function checks if 'output.json' exists, and if so, loads the data
    from the file. If the file does not exist, it returns an empty list.

    Returns
    -------
    list
        The loaded user data, or an empty list if the file does not exist.

    Examples
    --------
    >>> load_data_from_file()  # 'output.json' contains
    ... # [{'name': 'John Doe', 'age': '30'}]
    [{'name': 'John Doe', 'age': '30'}]

    >>> load_data_from_file()  # 'output.json' does not exist
    []
    """
    data: list = []
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            data.extend(json.load(file))
    return data


def display_menu() -> None:
    """
    Display the main menu options to the user.

    This function prints the available options to the console:
    1. Add user
    2. Update user
    3. Get user
    4. Exit

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


def process_choice(choice: str, data: list) -> list:
    """
    Process the user's choice and perform the corresponding action.

    This function takes the user's choice as input and calls the appropriate
    handler function based on the choice. If the choice is invalid, it logs
    an info message.

    Parameters
    ----------
    choice: str
        The user's choice (e.g., '1', '2', '3', '4').
    data: list
        The current list of user data.

    Returns
    -------
    list
        The updated list of user data, if applicable.

    Examples
    --------
    >>> data = []
    >>> process_choice('1', data)  # User inputs: 'Jane Smith', '25',
    ... # 'jane.smith@example.com', 'active'
    [{'name': 'Jane Smith', 'age': '25', 'email': 'jane.smith@example.com',
    'status': 'active'}]

    >>> process_choice('5', data)  # Invalid choice
    Invalid choice, try again
    """
    if choice == "1":
        return handle_add_user(data)
    elif choice == "2":
        return handle_update_user(data)
    elif choice == "3":
        handle_get_user(data)
    elif choice == "4":
        logging.info("Exiting...")
        exit()
    else:
        logging.info("Invalid choice, try again")
    return data


def main() -> None:
    """
    Executes the main functionality of the script.
    """
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)

    data: list = load_data_from_file()

    while True:
        display_menu()
        choice: str = input("Enter choice: ")
        data = process_choice(choice, data)


if __name__ == "__main__":
    main()
