# Standard library imports
import json
import os

# Third party imports
import pyfiglet

data: list[dict[str, str | int]] = []


def save_to_file(data_to_save: list[dict[str, str | int]]) -> None:
    """
    Save the provided data to a JSON file with indentation.

    Parameters
    ----------
    data_to_save: list[dict[str, str | int]]
        The data to be saved to the file.
    """
    with open("users.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(data_to_save, indent=4))


def add_user(name: str, age: str, email: str, status: str) -> None:
    """
    Add a user with the provided details and save to file.

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
    user: dict[str, str | int] = {
        "name": name,
        "age": age,
        "email": email,
        "status": status,
    }
    data.append(user)
    save_to_file(data)
    print(f"User added: {name}")


def update_user(name: str, field: str, value: str) -> None:
    """
    Update user information based on the provided name and field.

    Parameters
    ----------
    name: str
        The name of the user to update.
    field: str
        The field to update (name/age/email/status).
    value: str
        The new value for the specified field.
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
    Find and return user information based on the provided name.

    Parameters
    ----------
    name: str
        The name of the user to find.

    Returns
    -------
    dict[str, str | int] | str
        The user information if found, otherwise "Not found".
    """
    for user in data:
        if user["name"] == name:
            return user
    return "Not found"


def main() -> None:
    """Execute the main program logic."""
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
            age: str = input("Enter age: ")
            email: str = input("Enter email: ")
            status: str = input("Enter status (active/inactive): ")
            add_user(name, age, email, status)
        elif choice == "2":
            name: str = input("Enter name to update: ")
            field: str = input(
                "Enter field to update (name/age/email/status): ")
            value: str = input("Enter new value: ")
            update_user(name, field, value)
        elif choice == "3":
            name: str = input("Enter name to find: ")
            result = get_user(name)
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

        if os.path.exists("output.json"):
            with open("output.json", "r", encoding="utf-8") as file:
                data.clear()
                data.extend(json.load(file))


if __name__ == "__main__":
    main()
