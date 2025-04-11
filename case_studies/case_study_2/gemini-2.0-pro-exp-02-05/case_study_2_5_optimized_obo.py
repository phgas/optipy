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
from typing import Any, Dict, List, Union

import pyfiglet


logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


def save_to_file(data_to_save: List[Dict[str, Any]]) -> None:
    """
    Save the given data to a JSON file named 'users.json'.

    This function serializes the provided data and writes it to
    'users.json'. It uses an indent of 4 spaces for better readability.

    Parameters
    ----------
    data_to_save: List[Dict[str, Any]]
        The data to be saved to the file.

    Returns
    -------
    None

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30'}]
    >>> save_to_file(data)
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(
    name: str,
    age: str,
    email: str,
    status: str,
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add a new user to the data list and save it to a file.

    This function creates a new user dictionary with the given
    parameters, appends it to the data list, saves the updated list to
    'users.json', and logs an information message.

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
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    List[Dict[str, Any]]
        The updated list of user data with the new user added.

    Examples
    --------
    >>> data = []
    >>> data = add_user('Jane Smith', '25', 'jane.smith@example.com',
    ... 'active', data)
    >>> print(data)
    [{'name': 'Jane Smith', 'age': '25', 'email':
    'jane.smith@example.com', 'status': 'active'}]
    """
    user: Dict[str, Any] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    data.append(user)
    save_to_file(data)
    logging.info(f"User added: {name}")
    return data


def update_user(
    name: str,
    field: str,
    value: str,
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Update a specific field of an existing user in the data list and
    save it to a file.

    This function searches for a user by their name. If the user is
    found, the specified field is updated with the new value, the
    updated list is saved to 'users.json', and an information message
    is logged. If the user is not found, a message is logged.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str
        The new value for the specified field.
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    List[Dict[str, Any]]
        The updated list of user data.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> data = update_user('John Doe', 'age', '31', data)
    >>> print(data)
    [{'name': 'John Doe', 'age': '31', 'email':
    'john.doe@example.com', 'status': 'active'}]
    """
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            save_to_file(data)
            logging.info(f"Updated {name} {field}")
            return data
    logging.info("User not found")
    return data


def get_user(
    name: str, data: List[Dict[str, Any]]
) -> Union[Dict[str, Any], str]:
    """
    Retrieve a user's information by their name.

    This function searches for a user by their name in the provided
    data list. If the user is found, their information is returned. If
    not, 'Not found' is returned.

    Parameters
    ----------
    name: str
        The name of the user to retrieve.
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    Union[Dict[str, Any], str]
        The user's information as a dictionary if found, otherwise
        'Not found'.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> user = get_user('John Doe', data)
    >>> print(user)
    {'name': 'John Doe', 'age': '30', 'email': 'john.doe@example.com',
    'status': 'active'}

    >>> user = get_user('Jane Smith', data)
    >>> print(user)
    Not found
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def handle_add_user(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Handle the process of adding a new user by taking inputs.

    This function prompts the user to enter the new user's name, age,
    email, and status, then calls the 'add_user' function to add the
    user to the data list.

    Parameters
    ----------
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    List[Dict[str, Any]]
        The updated list of user data with the new user added.

    Examples
    --------
    >>> data = []
    >>> data = handle_add_user(data)
    Enter name: Jane Smith
    Enter age: 25
    Enter email: jane.smith@example.com
    Enter status (active/inactive): active
    """
    name: str = input("Enter name: ")
    age: str = input("Enter age: ")
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return add_user(name, age, email, status, data)


def handle_update_user(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Handle the process of updating an existing user by taking inputs.

    This function prompts the user to enter the name of the user to
    update, the field to update, and the new value, then calls the
    'update_user' function to update the user in the data list.

    Parameters
    ----------
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    List[Dict[str, Any]]
        The updated list of user data.

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> data = handle_update_user(data)
    Enter name to update: John Doe
    Enter field to update (name/age/email/status): age
    Enter new value: 31
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str = input("Enter new value: ")
    return update_user(name, field, value, data)


def handle_get_user(data: List[Dict[str, Any]]) -> None:
    """
    Handle the process of retrieving and displaying a user's
    information.

    This function prompts the user to enter the name of the user to
    find, then calls the 'get_user' function to retrieve the user's
    information. If the user is found, their details are logged;
    otherwise, a 'User not found' message is logged.

    Parameters
    ----------
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    None

    Examples
    --------
    >>> data = [{'name': 'John Doe', 'age': '30',
    ... 'email': 'john.doe@example.com', 'status': 'active'}]
    >>> handle_get_user(data)
    Enter name to find: John Doe
    INFO:root:Name: John Doe
    INFO:root:Age: 30
    INFO:root:Email: john.doe@example.com
    INFO:root:Status: active

    >>> handle_get_user(data)
    Enter name to find: Jane Smith
    INFO:root:User not found
    """
    name: str = input("Enter name to find: ")
    result: Union[Dict[str, Any], str] = get_user(name, data)
    if result == "Not found":
        logging.info("User not found")
    else:
        logging.info(f"Name: {result['name']}")
        logging.info(f"Age: {result['age']}")
        logging.info(f"Email: {result['email']}")
        logging.info(f"Status: {result['status']}")


def load_data() -> List[Dict[str, Any]]:
    """
    Load user data from 'output.json' if it exists.

    This function checks if 'output.json' exists. If it does, the
    function reads the JSON data from the file and returns it as a list
    of dictionaries. If the file does not exist, it returns an empty
    list.

    Parameters
    ----------
    None

    Returns
    -------
    List[Dict[str, Any]]
        The loaded user data, or an empty list if 'output.json' does
        not exist.

    Examples
    --------
    >>> # Assuming 'output.json' contains:
    ... # [{'name': 'John Doe', 'age': '30'}]
    >>> data = load_data()
    >>> print(data)
    [{'name': 'John Doe', 'age': '30'}]

    >>> # If 'output.json' does not exist
    >>> data = load_data()
    >>> print(data)
    []
    """
    data: List[Dict[str, Any]] = []
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            data.extend(json.load(file))
    return data


def display_menu() -> None:
    """
    Display the main menu options to the user.

    This function logs the menu options for adding, updating, getting a
    user, and exiting the program.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> display_menu()
    INFO:root:1. Add user
    INFO:root:2. Update user
    INFO:root:3. Get user
    INFO:root:4. Exit
    """
    logging.info("1. Add user")
    logging.info("2. Update user")
    logging.info("3. Get user")
    logging.info("4. Exit")


def get_choice() -> str:
    """
    Prompt the user to enter their choice and return it.

    This function takes user input for their choice of action in the
    menu.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The user's choice as a string.

    Examples
    --------
    >>> choice = get_choice()
    Enter choice: 1
    >>> print(choice)
    1
    """
    return input("Enter choice: ")


def process_choice(
    choice: str, data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process the user's choice and execute the corresponding action.

    This function takes the user's choice and the current user data
    list. Depending on the choice, it either calls the appropriate
    handler function (for adding, updating, or getting a user) or exits
    the program. If the choice is invalid, it logs an error message.

    Parameters
    ----------
    choice: str
        The user's choice.
    data: List[Dict[str, Any]]
        The current list of user data.

    Returns
    -------
    List[Dict[str, Any]]
        The updated list of user data after performing the chosen
        action.

    Examples
    --------
    >>> data = []
    >>> data = process_choice('1', data)
    Enter name: John Doe
    Enter age: 30
    Enter email: john.doe@example.com
    Enter status (active/inactive): active

    >>> data = process_choice('2', data)
    Enter name to update: John Doe
    Enter field to update (name/age/email/status): age
    Enter new value: 31

    >>> process_choice('3', data)
    Enter name to find: John Doe
    INFO:root:Name: John Doe
    INFO:root:Age: 31
    INFO:root:Email: john.doe@example.com
    INFO:root:Status: active

    >>> data = process_choice('4', data)
    INFO:root:Exiting...

    >>> data = process_choice('5', data)
    INFO:root:Invalid choice, try again
    """
    if choice == "1":
        return handle_add_user(data)
    elif choice == "2":
        return handle_update_user(data)
    elif choice == "3":
        handle_get_user(data)
    elif choice == "4":
        logging.info("Exiting...")
        return data
    else:
        logging.info("Invalid choice, try again")
    return data


def main() -> None:
    """
    Execute the main functionality of the script.

    This function sets up the application by displaying a header and
    loading existing user data. It then enters a loop where it displays
    the menu, takes user input for their choice, and processes that
    choice. The loop continues until the user chooses to exit (choice
    '4').

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> main()
    INFO:root:     _   __      _    ____
    INFO:root:    / | / /___  (_)  / __ \____  ____  ____ _
    INFO:root:   /  |/ / __ \/ /  / / / / __ \/ __ \/ __ `/
    INFO:root:  / /|  / /_/ / /  / /_/ / /_/ / / / / /_/ /
    INFO:root: /_/ |_/\____/_/  /_____/\____/_/ /_/\__, /
    INFO:root:                                     /____/
    INFO:root:1. Add user
    INFO:root:2. Update user
    INFO:root:3. Get user
    INFO:root:4. Exit
    Enter choice: 1
    ...
    """
    header: str = pyfiglet.figlet_format("User Manager")
    logging.info(header)
    data: List[Dict[str, Any]] = load_data()

    while True:
        display_menu()
        choice: str = get_choice()
        data = process_choice(choice, data)
        if choice == "4":
            break


if __name__ == "__main__":
    main()
