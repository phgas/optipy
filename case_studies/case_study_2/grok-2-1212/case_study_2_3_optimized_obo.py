"""
User Manager Module
==================

A module for managing user data, including adding, updating, and retrieving user information.

This module provides functionality to interact with a JSON file to store and manage user data. It includes
functions for adding new users, updating existing user information, retrieving user details, and a simple
command-line interface for user interaction.

Examples
--------
>>> from user_manager import add_user, get_user
>>> data = add_user("John Doe", 30, "john@example.com", "active")
>>> user = get_user("John Doe")
>>> print(user['email'])
john@example.com
"""

# Standard library imports
import json
import logging
import os

# Third party imports

# Local imports

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
    >>> data = [{"name": "John", "age": 30}]
    >>> save_to_file(data)
    """
    with open("users.json", "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)


def add_user(name: str, age: int, email: str, status: str) -> list[dict[str, str | int]]:
    """
    Add a new user to the data and save it.

    This function creates a new user dictionary with the provided details, adds it to the existing data,
    saves the updated data to the file, and logs the addition.

    Parameters
    ----------
    name: str
        The name of the user.
    age: int
        The age of the user.
    email: str
        The email address of the user.
    status: str
        The status of the user, typically 'active' or 'inactive'.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of user data after adding the new user.

    Examples
    --------
    >>> data = add_user("Jane Doe", 25, "jane@example.com", "active")
    >>> print(data[-1]['name'])
    Jane Doe
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
    return data


def update_user(name: str, field: str, value: str | int) -> list[dict[str, str | int]]:
    """
    Update a user's information in the data and save it.

    This function searches for a user by name, updates the specified field with the new value, saves the
    updated data to the file, and logs the update. If the user is not found, it logs a message.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (e.g., 'name', 'age', 'email', 'status').
    value: str | int
        The new value for the specified field.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of user data after the update.

    Examples
    --------
    >>> data = update_user("John Doe", "email", "john.new@example.com")
    >>> print(data[0]['email'])
    john.new@example.com
    """
    data: list[dict[str, str | int]] = load_data()
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            save_to_file(data)
            logging.info(f"Updated {name} {field}")
            return data
    logging.info("User not found")
    return data


def get_user(name: str) -> dict[str, str | int] | str:
    """
    Retrieve a user's information by name.

    This function searches for a user by name in the data and returns the user's details if found. If the
    user is not found, it returns a 'Not found' string.

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
    >>> print(user['email'] if isinstance(user, dict) else user)
    john@example.com
    """
    data: list[dict[str, str | int]] = load_data()
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> list[dict[str, str | int]]:
    """
    Load user data from the JSON file.

    This function reads the 'users.json' file if it exists and returns the data as a list of dictionaries.
    If the file does not exist, it returns an empty list.

    Returns
    -------
    list[dict[str, str | int]]
        A list of dictionaries containing user data, or an empty list if the file does not exist.

    Examples
    --------
    >>> data = load_data()
    >>> print(len(data))
    0
    """
    if os.path.exists("users.json"):
        with open("users.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def display_menu() -> str:
    """
    Display the menu options and get user input.

    This function prints the menu options for the user to choose from and returns the user's choice.

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


def process_add_user() -> list[dict[str, str | int]]:
    """
    Process the addition of a new user.

    This function prompts the user for details and calls the add_user function to add the new user.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of user data after adding the new user.

    Examples
    --------
    >>> data = process_add_user()
    >>> print(data[-1]['name'])
    New User
    """
    name: str = input("Enter name: ")
    age: int = int(input("Enter age: "))
    email: str = input("Enter email: ")
    status: str = input("Enter status (active/inactive): ")
    return add_user(name, age, email, status)


def process_update_user() -> list[dict[str, str | int]]:
    """
    Process the update of an existing user.

    This function prompts the user for the name of the user to update, the field to update, and the new value,
    then calls the update_user function to perform the update.

    Returns
    -------
    list[dict[str, str | int]]
        The updated list of user data after the update.

    Examples
    --------
    >>> data = process_update_user()
    >>> print(data[0]['email'])
    updated@example.com
    """
    name: str = input("Enter name to update: ")
    field: str = input("Enter field to update (name/age/email/status): ")
    value: str | int = input("Enter new value: ")
    if field == "age":
        value = int(value)
    return update_user(name, field, value)


def process_get_user() -> None:
    """
    Process the retrieval of a user's information.

    This function prompts the user for a name, retrieves the user's details, and logs the information if found.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> process_get_user()
    """
    name: str = input("Enter name to find: ")
    result: dict[str, str | int] | str = get_user(name)
    if result == "Not found":
        logging.info("User not found")
    else:
        logging.info(f"Name: {result['name']}")
        logging.info(f"Age: {result['age']}")
        logging.info(f"Email: {result['email']}")
        logging.info(f"Status: {result['status']}")


def process_exit() -> None:
    """
    Process the exit command.

    This function logs an exit message.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> process_exit()
    """
    logging.info("Exiting...")


def process_choice(choice: str) -> str | list[dict[str, str | int]] | None:
    """
    Process the user's menu choice.

    This function calls the appropriate function based on the user's choice and returns the result.

    Parameters
    ----------
    choice: str
        The user's menu choice as a string.

    Returns
    -------
    str | list[dict[str, str | int]] | None
        The result of the processed choice, which can be a string, a list of user data, or None.

    Examples
    --------
    >>> result = process_choice("1")
    >>> print(result[-1]['name'])
    New User
    """
    if choice == "1":
        return process_add_user()
    elif choice == "2":
        return process_update_user()
    elif choice == "3":
        process_get_user()
    elif choice == "4":
        process_exit()
        return "exit"
    else:
        logging.info("Invalid choice, try again")
    return None


def main() -> None:
    """
    Execute the main functionality of the script.

    This function runs the user manager application, displaying a menu and processing user choices until the user chooses to exit.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> main()
    """
    import pyfiglet
    header: str = pyfiglet.figlet_format("User Manager")
    print(header)
    data: list[dict[str, str | int]] = load_data()
    while True:
        choice: str = display_menu()
        result: str | list[dict[str, str | int]
                           ] | None = process_choice(choice)
        if result == "exit":
            break
        if result:
            data = result


if __name__ == "__main__":
    main()
