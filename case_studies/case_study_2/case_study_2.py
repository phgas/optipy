import os, json, pyfiglet

data = []

def saveToFile(data_to_save):
    f = open("users.json", "w")
    f.write(json.dumps(data_to_save, indent=4))
    f.close()

def addUser(name, age, email, status):
    # adding a user with all details
    user = {}
    user["name"] = name
    user["age"] = age
    user["email"] = email
    user["status"] = status
    data.append(user)
    saveToFile(data)
    print("User added: " + name)

def updateUser(name, field, value):
    # update user info based on name
    found = 0
    for i in range(len(data)):
        if data[i]["name"] == name:
            data[i][field] = value
            found = 1
    if found == 0:
        print("User not found")
    else:
        saveToFile(data)
        print("Updated " + name + " " + field)

def getUser(name):
    # find and return user
    for i in range(len(data)):
        if data[i]["name"] == name:
            return data[i]
    return "Not found"

def main():
    # main program logic
    header = pyfiglet.figlet_format("User Manager")
    print(header)
    while True:
        print("1. Add user")
        print("2. Update user")
        print("3. Get user")
        print("4. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter name: ")
            age = input("Enter age: ")
            email = input("Enter email: ")
            status = input("Enter status (active/inactive): ")
            addUser(name, age, email, status)
        elif choice == "2":
            name = input("Enter name to update: ")
            field = input("Enter field to update (name/age/email/status): ")
            value = input("Enter new value: ")
            updateUser(name, field, value)
        elif choice == "3":
            name = input("Enter name to find: ")
            result = getUser(name)
            if result == "Not found":
                print("User not found")
            else:
                print("Name: " + result["name"])
                print("Age: " + result["age"])
                print("Email: " + result["email"])
                print("Status: " + result["status"])
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice, try again")

        # Check if file exists and load it
        if os.path.exists("output.json"):
            f = open("output.json", "r")
            data.clear()
            data.extend(json.load(f))
            f.close()

# Start the program
main()