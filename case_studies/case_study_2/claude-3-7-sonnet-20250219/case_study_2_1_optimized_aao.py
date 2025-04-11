# Standard library imports
import json
import os

# Third party imports
import pyfiglet

import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)

data: list[dict[str, str]] = []


def save_to_file(data_to_save: list[dict[str, str]]) -> None:
    """
    Save user data to a JSON file.

    Parameters
    ----------
    data_to_save: list[dict[str, str]]
        The user data to save.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(name: str, age: str, email: str, status: str) -> None:
    """
    Add a new user to the data list and save to file.

    Parameters
    ----------
    name: str
        The name of the user.
    age: str
        The age of the user.
    email: str
        The email of the user.
    status: str
        The status of the user (active/inactive).
    """
    user: dict[str, str] = {}
    user["name"] = name
    user["age"] = age
    user["email"] = email
    user["status"] = status
    data.append(user)
    save_to_file(data)
    logging.info(f"User added: {name}")


def update_user(name: str, field: str, value: str) -> None:
    """
    Update a user's information based on their name.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str
        The new value for the field.
    """
    found = False
    for i, user in enumerate(data):
        if user["name"] == name:
            data[i][field] = value
            found = True

    if not found:
        logging.info("User not found")
    else:
        save_to_file(data)
        logging.info(f"Updated {name} {field}")


def get_user(name: str) -> dict[str, str] | str:
    """
    Find and return a user by name.

    Parameters
    ----------
    name: str
        The name of the user to find.

    Returns
    -------
    dict[str, str] | str
        The user data if found, or "Not found" if not found.
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def load_data() -> None:
    """
    Load user data from file if it exists.
    """
    if os.path.exists("output.json"):
        with open("output.json", "r", encoding="utf-8") as file:
            data.clear()
            data.extend(json.load(file))


def display_user(user: dict[str, str]) -> None:
    """
    Display user information.

    Parameters
    ----------
    user: dict[str, str]
        The user data to display.
    """
    logging.info(f"Name: {user['name']}")
    logging.info(f"Age: {user['age']}")
    logging.info(f"Email: {user['email']}")
    logging.info(f"Status: {user['status']}")


def get_user_input() -> tuple[str, str, str, str]:
    """
    Get user input for adding a new user.

    Returns
    -------
    tuple[str, str, str, str]
        The name, age, email, and status of the new user.
    """
    name = input("Enter name: ")
    age = input("Enter age: ")
    email = input("Enter email: ")
    status = input("Enter status (active/inactive): ")
    return name, age, email, status


def get_update_info() -> tuple[str, str, str]:
    """
    Get user input for updating a user.

    Returns
    -------
    tuple[str, str, str]
        The name, field, and value for the update.
    """
    name = input("Enter name to update: ")
    field = input("Enter field to update (name/age/email/status): ")
    value = input("Enter new value: ")
    return name, field, value


def main() -> None:
    """
    Executes the main functionality of the script.
    """
    header = pyfiglet.figlet_format("User Manager")
    print(header)

    while True:
        print("1. Add user")
        print("2. Update user")
        print("3. Get user")
        print("4. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            name, age, email, status = get_user_input()
            add_user(name, age, email, status)
        elif choice == "2":
            name, field, value = get_update_info()
            update_user(name, field, value)
        elif choice == "3":
            name = input("Enter name to find: ")
            result = get_user(name)
            if result == "Not found":
                logging.info("User not found")
            else:
                display_user(result)
        elif choice == "4":
            logging.info("Exiting...")
            break
        else:
            logging.info("Invalid choice, try again")

        load_data()


if __name__ == "__main__":
    main()
