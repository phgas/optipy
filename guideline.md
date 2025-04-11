# Clean Code Guideline

##  1. General Pythonic Practices

### f-Strings
  - Use f-Strings instead of String Concatenation/filter()
    ```python
    # WRONG
    print(name + " is " + str(age) + " years old.")
    ```
    ```python
    # BETTER
    print("{} is {} years old.".format(name, age))
    ```
    ```python
    # BEST
    print(f"{name} is {age} years old.")
    ```

### Use Logging instead of print()-Statements
  - Define the logging.basicConfig in the module level (at the top of the file, after imports) and not within `if __name__ == "__main__":` block.
    ```python
    # WRONG
    def divide(dividend, divisor):
        print(f"Received input: dividend={dividend}, divisor={divisor}")

        if divisor == 0:
            print("Error: Division by zero attempted!")
            raise ValueError("Division by zero is not allowed.")

        result = dividend / divisor
        print(f"Calculation successful: {dividend} / {divisor} = {result}")
        return result
    ```
    ```python
    # CORRECT
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(asctime)s.%(msecs)03d][%(levelname)s]"
            "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
        ),
        datefmt="%d-%m-%Y %H:%M:%S",
    )


    def divide(dividend, divisor):
        logging.info(f"Received input: {dividend=}, {divisor=}")

        if divisor == 0:
            logging.error("Error: Division by zero attempted!")
            raise ValueError("Division by zero is not allowed.")

        result = dividend / divisor
        logging.info(f"Calculation successful: {dividend} / {divisor} = {result}")
        return result
    ```
  - Use f-strings within the logging command, not %s.

### List Comprehensions
  - Simplify for-Loops with List Comprehensions
    ```python
    # WRONG
    even_numbers = []
    for number in numbers:
        if number % 2 == 0:
            even_numbers.append(number)
    ```
    ```python
    # CORRECT
    even_numbers = [number for number in numbers if number % 2 == 0]
    ```

### Context Managers
  - Using Context Managers when accessing files
    ```python
    # WRONG
    input_file = open("input.txt", "r")
    content = input_file.read()
    input_file.close()
    ```
    ```python
    # BETTER
    with open("input.txt", "r") as input_file:
        content = input_file.read()
    ```

  - Always specify encoding to avoid decoding issues
    ```python
    # BEST
    with open("input.txt", "r", encoding="utf-8") as input_file:
        content = input_file.read()
    ```
    
### When accessing an index use `enumerate(elements)` instead of `range(len(elements))`
    ```python
    # WRONG
    for i in range(len(elements)):
        print(f"Index: {i}, Element: {elements[i]}")
    ```
    ```python
    # CORRECT
    for i, element in enumerate(elements):
        print(f"Index: {i}, Element: {element}")
    ```


##  2. Refactoring

### Avoid deep nesting by refactoring code into functions/methods with a maximum of 3 indentation levels
    ```python
    # WRONG
    def check_permissions(user):
        if user:
            if user.is_valid():
                if user.has_role("admin"):
                    if user.has_permission("edit"):
                        print("Access granted!")
                    else:
                        print("Permission denied!")
                else:
                    print("User has no admin role!")
            else:
                print("User is invalid!")
        else:
            print("No user provided!")
    ```
    ```python
    # CORRECT
    def check_permissions(user):
        if not user:
            print("No user provided!")
            return
        
        if not user.is_valid():
            print("User is invalid!")
            return

        if not user.has_role("admin"):
            print("User has no admin role!")
            return

        if not user.has_permission("edit"):
            print("Permission denied!")
            return
        
        print("Access granted!")
    ```

### Refactor long functions into smaller ones with a maximum line count of 10 
  - Reduce complexity & Improve maintainability
    ```python
    # WRONG
    def process_order(order):
        total = sum(item['price'] * item['quantity'] for item in order['items'])
        
        if total > 100:
            discount = total * 0.1
        else:
            discount = 0

        final_price = total - discount
        
        print(f"Final Price: {final_price}")
    ```
    ```python
    # CORRECT
    def calculate_total(order):
        return sum(item['price'] * item['quantity'] for item in order['items'])

    def apply_discount(total):
        return total * 0.1 if total > 100 else 0

    def process_order(order):
        total = calculate_total(order)
        discount = apply_discount(total)
        final_price = total - discount
        print(f"Final Price: {final_price}")
    ```

### One Statement per line
  - This accounts for assignments too:
    ```python
    # WRONG
    customer_name = "Albert Einstein", customer_id = 1234
    ```
    ```python
    # CORRECT
    customer_name = "Albert Einstein"
    customer_id = 1234
    ```

  - This accounts for bool-assignments and if-statements:
    ```python
    # WRONG
    if number > 0 and number % 2 == 0 and number % 99 == 0: print(f"{number} is valid!")
    ```
    ```python
    # CORRECT
    is_positive = number > 0  
    is_even = number % 2 == 0  
    is_divisible_by_ninetynine = number % 99 == 0  

    if is_positive and is_even and is_divisible_by_ninetynine:
        print(f"{number} is valid!")  
    ```

### Remove duplicate code/functions/methods/classes
    ```python
    # WRONG
    print("Good morning!")
    print("How's your mood right now?")
    mood = input()
    print(f"Glad to hear that you feel {mood}.")

    print("Good afternoon!")
    print("How's your mood right now?")
    mood = input()
    print(f"Glad to hear that you feel {mood}.")

    print("Good evening!")
    print("How's your mood right now?")
    mood = input()
    print(f"Glad to hear that you feel {mood}.")
    ```
    ```python
    # BETTER
    def check_mood():
        print("How's your mood right now?")
        mood = input()
        print(f"Glad to hear that you feel {mood}.")

    print("Good morning!")
    check_mood()

    print("Good afternoon!")
    check_mood()

    print("Good evening!")
    check_mood()
    ```
    ```python
    # BEST
    def check_mood(time_of_day):
        print(f"Good {time_of_day}!")
        print("How's your mood right now?")
        mood = input()
        print(f"Glad to hear that you feel {mood}.")


    times_of_day = ["morning", "afternoon", "evening"]
    for time_of_day in times_of_day:
        check_mood(time_of_day)
    ```

### Remove dead/commented out code
  - Remove dead code commented out with Comments (#)
    ```python
    # WRONG
    def calculate_total(price, discount):
        # total = price + (price * discount)
        return price * (1 + discount)
    ```
    ```python
    # CORRECT
    def calculate_total(price, discount):
        return price * (1 - discount)
    ```

### Refactor the code to ensure no global variables are used
    ```python
    # WRONG
    def read_api_key() -> None:
        load_dotenv(override=True)
        global API_KEY
        API_KEY = os.getenv("API_KEY")
    ```
    ```python
    # CORRECT
    def read_api_key() -> str:
        load_dotenv(override=True)
        API_KEY = os.getenv("API_KEY")
        return API_KEY
    ```

### Remove unnecessary comments (except module categories `# Standard library imports`, `# Third party imports`, `# Local imports`)
  - ```python
    # WRONG
    original_price = 1000  # Original price of the product
    discount_rate = 0.1  # Discount rate (0.1 = 10%)

    # Calculate the discounted price for the current year
    discounted_price = original_price * (1 - discount_rate)

    # Print the discounted price
    print(f"{discounted_price}$")
    ```
    ```python
    # CORRECT
    original_price = 1000
    discount_rate = 0.1

    discounted_price = original_price * (1 - discount_rate)
    print(f"{discounted_price}$")
    ```

##  3. Naming Conventions
- APPLY all following naming conventions
- Make sure no variable is defined twice to avoid type hinting issues

### General Naming Rules (recommendation by PEP-8)
- All words should be english and consist of ASCII letters (no accent marks).
    ```python
    # WRONG
    öffnungszeiten = {...}
    straße = "..."
    ```
    ```python 
    # CORRECT
    opening_hours = {...}
    street = "..."
    ```

- Use short, common names for clarity.
    ```python 
    # WRONG
    def retrieve_data():
        ...
    ```
    ```python
    # CORRECT
    def get_data():
        ...
    ```
  
- Pick one term and use it consistently (e.g. `data`).
    ```python
    # WRONG
    def get_data():
        # ...

    def process_info():
        # ...

    def save_content():
        # ...
    ```
    ```python 
    # CORRECT
    def get_data():
        # ...

    def process_data():
        # ...

    def save_data():
        # ...
    ```

- Avoid meaningless names & uncommon abbreviations (If necessary, use abbrevations, when longer than 20 characters, e.g. `very_very_long_string` → `very_very_long_str`)
    ```python
    # WRONG
    cstmrs = [...]
    nbrs = [...]
    temp_var = {...}
    ```
    ```python 
    # CORRECT
    customers = [...]
    numbers = [...]
    temperature_variance = {...}
    ```

- Use descriptive names to clarify functionality.
  ```python
  # WRONG
  def f(...):
    # ...
  ```
  ```python 
  # CORRECT
  def eur_to_usd(...):
    # ...
  ```

- Avoid ambiguous names by specifying context.
  ```python
  # WRONG
  def euro_to_dollar(...):
    # ...
  ```
  ```python 
  # CORRECT
  def eur_to_usd(...):
    # ...
  ```

- Generally avoid appending types to names unless necessary for clarity.
    ```python
    # WRONG
    customer_list = [...]
    country_to_currency_dict = {...}
    PI_value = 3.14
    ```
    ```python 
    # CORRECT
    customers = [...]
    country_to_currency = {...}
    PI = 3.14
    ```
    Except: In some cases it makes sense to append variable type, but only when differentiating between multiple representations of the same data.
    _str, _int, _float, _list, _dict, _set
    ```python 
    # CORRECT
    age_int = 33
    age_str = str(age_int)
    ```
- Don’t Overwrite Built-in Names: ArithmeticError, AssertionError, AttributeError, BaseException, BaseExceptionGroup, BlockingIOError, BrokenPipeError, BufferError, BytesWarning, ChildProcessError, ConnectionAbortedError, ConnectionError, ConnectionRefusedError, ConnectionResetError, DeprecationWarning, EOFError, Ellipsis, EncodingWarning, EnvironmentError, Exception, ExceptionGroup, False, FileExistsError, FileNotFoundError, FloatingPointError, FutureWarning, GeneratorExit, IOError, ImportError, ImportWarning, IndentationError, IndexError, InterruptedError, IsADirectoryError, KeyError, KeyboardInterrupt, LookupError, MemoryError, ModuleNotFoundError, NameError, None, NotADirectoryError, NotImplemented, NotImplementedError, OSError, OverflowError, PendingDeprecationWarning, PermissionError, ProcessLookupError, RecursionError, ReferenceError, ResourceWarning, RuntimeError, RuntimeWarning, StopAsyncIteration, StopIteration, SyntaxError, SyntaxWarning, SystemError, SystemExit, TabError, TimeoutError, True, TypeError, UnboundLocalError, UnicodeDecodeError, UnicodeEncodeError, UnicodeError, UnicodeTranslateError, UnicodeWarning, UserWarning, ValueError, Warning, WindowsError, ZeroDivisionError, __build_class__, __debug__, __doc__, __import__, __loader__, __name__, __package__, __spec__, abs, aiter, all, anext, any, ascii, bin, bool, breakpoint, bytearray, bytes, callable, chr, classmethod, compile, complex, copyright, credits, delattr, dict, dir, divmod, enumerate, eval, exec, exit, filter, float, format, frozenset, getattr, globals, hasattr, hash, help, hex, id, input, int, isinstance, issubclass, iter, len, license, list, locals, map, max, memoryview, min, next, object, oct, open, ord, pow, print, property, quit, range, repr, reversed, round, set, setattr, slice, sorted, staticmethod, str, sum, super, tuple, type, vars, zip
    ```python 
    # WRONG
    class Order:
        def __init__(self, id):
            self.id = id
    ```
    ```python 
    # CORRECT
    class Order:
        def __init__(self, id_):
            self.id_ = id_
    ```

### Naming Modules
- Modules should be written in lowercase snake_case
- If necessary, use underscores to increase readability. 
    ```python
    # WRONG
    import SomeModule
    import someModule
    import Somemodule
    import somemodule
    import Some_Module
    import some_Module
    import Some_module
    ```
    ```python
    # CORRECT
    import module
    import some_other_module
    ```

### Naming Constants
  - Constant variables should be written in uppercase SNAKE_CASE.
  - Use precisely named constants to avoid magic numbers.
  ```python
  # WRONG
  profit_in_usd = 0.9 * profit_in_eur
  ```
  ```python 
  # CORRECT
  EUR_TO_USD_RATE = 0.9
  profit_in_usd = EUR_TO_USD_RATE * profit_in_eur
  ```

### Naming Integer variables
- Integer variables should be written in lowercase snake_case.
- Most common: `number_of_<something>` or `<something>_count` (e.g. `number_of_failures` or `failure_count`)
- If unit not self-evident, add it at the end (e.g. `retry_delay` → `retry_delay_in_ms`)

    ```python
    # WRONG
    failures = 10
    retry_delay = 500
    distance = 33
    price = 33
    ```
    ```python 
    # CORRECT
    failure_count = 10
    retry_delay_in_ms = 500
    distance_in_km = 33
    price_in_eur = 33
    ```

### Naming Float variables
- Float variables should be written in lowercase snake_case.
- Most common: `<something>_amount` (e.g. `rainfall_amount`), use Int for money to avoid rounding errors.
- If unit not self-evident, add it at the end
  - `rainfall_amount_in_mm`
  - `angle_in_degrees` (values 0-360)
  - `failure_percent` (values 0-100)
  - `failure_ratio` (values 0-1)

### Naming Boolean variables
- Boolean variables should be written in lowercase snake_case.
- The verb should be in the middle of boolean variable.
- Extract Constant for Boolean Expression
- Most common: 
  - `is_<something>` → e.g. `is_disabled`
  - `has_<something>` → e.g. `has_errors`
  - `did_<something>` → e.g. `did_update`
  - `should_<something>` → e.g. `should_update`
  - `will_<something>` → e.g. `will_update`
- **No-Go: `<passive-verb>_something` (f.e.: `inserted_field` → `field_was_insterted` verb should be in the middle of boolean)**

    ```python
    # WRONG
    if is_pool_full:
        # ...
    ```
    ```python 
    # CORRECT
    if pool_is_full:
        # ...
    ```
    ---
    ```python
    # WRONG
    if inserted_table:
        # ...
    ```
    ```python 
    # CORRECT
    if table_was_insterted:
        # ...
    ```
    ---
    ```python
    # WRONG
    machine_was_started = machine.start()
    if not machine_was_started:
        # ...
    ```
    ```python 
    # CORRECT
    machine_was_not_started = not machine.start()
    if machine_was_not_started:
        # ...
    ```
  ---
- Extract boolean variables from constants, so if-statement only consist of explicit boolean variables
    ```python
    # WRONG
    if person.age > 18 and person.has_license:
        is_allowed_to_drive = True
    ```
    ```python 
    # CORRECT
    is_adult = person.age > 18
    has_license = person.has_license
    is_allowed_to_drive = is_adult and has_license
    ```

### Naming String variables
- String variables should be written in lowercase snake_case.
    ```python
    # WRONG
    UserName = "Amadeus"
    ```
    ```python 
    # CORRECT
    user_name = "Amadeus"
    ```
    - Consistent usage of quotation marks
    ```python
    # WRONG
    user_names = ["Amadeus", 'Ludwig']
    ```
    ```python 
    # CORRECT
    user_names = ["Amadeus", "Ludwig"]
    ```

### Naming Collection (List and Set) Variables
- Collection variables should be written in lowercase snake_case.
- Use plural form of noun for lists and sets (e.g. customers, tasks)
- Avoid embedding type (e.g., customer_list, task_set) :
    ```python
    # WRONG
    customer_list = [...]
    student_id_set = {"001", "002", "003"}
    ```
    ```python 
    # CORRECT
    customers = [...]
    student_ids = {"001", "002", "003"}
    ```

    unless necessary in some edge cases (e.g. queue_of_tasks) or in the following example:
    ```python
    fruits_string = "banana apple cherry banana"
    fruits_list = fruits_string.split(" ")
    fruits_set = set(fruits_list)
    ```
  
### Naming Dictionary Variables
- Dictionary variables should be written in lowercase snake_case 
- Dictionary variables should be follow the pattern `key_to_value` or `key_to_values`
    ```python
    country_currencies = {
        "AT": "EUR",
        ...
    }
    ```
    ```python
    # CORRECT
    country_to_currency = {
        "AT": "EUR",
        ...
    }
    ```
    ----
    ```python
    # WRONG
    country_currencies = {
        "AT": ["EUR", "USD"],
        ...
    }
    ```
    ```python
    # CORRECT
    country_to_currencies = {
        "AT": ["EUR", "USD"],
        ...
    }
        ```

### Naming Pair and Tuple Variables
- Tuples variables should be written in lowercase snake_case.
- If variable not needed use throwaway variable `_`.
  ```python
  # CORRECT
  height_and_width_in_mm = (100, 200)
  coordinates = (48.2394, 16.3778)
  ```
- If variable not needed use throwaway variable `_`.
  ```python
  # CORRECT
  locations = [
      ("Vienna", 48.2394, 16.3778),
  ]

  for city, _, _ in locations:
      print(city)
  ```

### Naming Classes
- Class names should be written in PascalCase.
- Private attributes & methods should always begin with an underscore _.
- Public attributes & methods should never begin with an underscore _.
    ```python
    # CORRECT
    class BankAccount:
        def __init__(self, owner, balance):
            self.owner = owner
            self._balance = balance

        def get_balance(self):
            return self._balance
            
        def deposit(self, amount):
            if self._is_valid_amount(amount):
                self._balance += amount

        def _is_valid_amount(self, amount):
            return amount > 0
    ```
- The first argument for class methods should always be named cls in lowercase.
  ```python
  # CORRECT
  class Machine:
      count = 0

      def __init__(self, brand, model):
          self.brand = brand
          self.model = model
          Machine.count += 1

      @classmethod
      def get_count(cls):
          return cls.count
    ```

- Class attributes should not repeat class name:
  ```python
  # WRONG
  class Order:
    def __init__(self, order_id, order_state):
        self.order_id = order_id
        self.order_state = order_state
  ```
  ```python
  # CORRECT
  class Order:
    def __init__(self, id_, state):
        self.id_ = id_
        self.state = state
  ```

### Naming Object Variables
- Object variables should be written in lowercase snake_case.
- Object names should reflect the underlying class and may include adjectives or descriptive terms to clarify their role or function.
  ```python
  # CORRECT
  class Task:
      def __init__(self, name):
          self.name = name

  build_task = Task("Build")
  ```
- For general concepts, including the class name in the object name may not be possible. Instead, the object's role should be clear from its name and context.
  ```python
  # CORRECT
  class Fruit:
      def __init__(self, name, color):
          self.name = name
          self.color = color

  apple = Fruit("Apple", "Red")
  banana = Fruit("Banana", "Yellow")
  ```

### Naming Functions, method
- Function and method names should be written in lowercase snake_case.
- The first argument for methods should always be named self in lowercase.


##  4. Use Main Function and Main Name Idiom
- Always define a `main()` function to encapsulate the main logic of your script (put each step of the main logic into a seperate function).
- Use the `if __name__ == "__main__":` idiom to ensure the script executes correctly when run directly, but not when imported.
  ```python
  # WRONG
  print("Hello World!")
  ```
  ```python
  # BETTER
  def main() -> None:
    print("Hello World!")

  if __name__ == "__main__":
    main()
  ```

- Make sure the `main()` function is always the last function.
  ```python
  # WRONG
    def main() -> None:
        ...

    def some_function() -> None:
        ...
  ```
  ```python
  # CORRECT
    def some_function() -> None:
        ...
        
    def main() -> None:
        ...
  ```

- Each step in the `main()`-function should be encapsulated within a dedicated function
    ```python
    # WRONG
    def square(number):
        return number**2


    def save_result(number, squared):
        with open("results.txt", "w") as file:
            file.write(f"The square of {number} is {squared}\n")
        print("Result saved to results.txt")


    def main():
        while True:
            try:
                number = int(input("Enter a number to be squared: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number (int).")
        squared = square(number)
        save_result(number, squared)
    ```
    ```python
    # CORRECT
    def get_number():
        while True:
            try:
                return int(input("Enter a number to be squared: "))
            except ValueError:
                print("Invalid input. Please enter a valid number (int).")


    def square(number):
        return number**2


    def save_result(number, squared):
        with open("results.txt", "w") as file:
            file.write(f"The square of {number} is {squared}\n")
        print("Result saved to results.txt")


    def main():
        number = get_number()
        squared = square(number)
        save_result(number, squared)
    ```

- Eventough the python community debates about, whether adding a docstring to the main() function is necessary, it increases clarity for unexperienced developers.
  ```python
  # BEST
  def main() -> None:
    """Executes the main functionality of the script."""
    print("Hello World!")

  if __name__ == "__main__":
    main()
  ```
  
- Never put the defintion of a class/function  in the `main()` function. It should only demonstrate the main logic/functionality.
  ```python
  # WRONG
    def main() -> None:
        """Executes the main functionality of the script."""
        class Machine:
            ...
  ```
  ```python
  # CORRECT
    class Machine:
        ...
    def main() -> None:
        """Executes the main functionality of the script."""
        machine = Machine(...)
  ```

##  5. Use Type Annotations/Hinting
- Keep logging as is (no type hints needed).
- Do not use Type Hints for global variables
- Use type hints to keep your code clean and intuitive.
  ```python
  variable: type = value
  ```
  ```python
  def function_name(param1: param1_type, param2: param2_type) -> return_type:
  ```
- Assign type hints to variables
  ```python
  # CORRECT - Python 3.10+
  name: str = "Amadeus"
  age: int = 66
  success_rate: float = 0.95
  machine_has_stopped: bool = False
  payload: bytes = b'Sensor_Data'
  last_name_to_age: dict[str, int] = {"Mozart": 66, "Beethoven": 99}
  grades: list[int] = [1, 4, 2, 2, 3]
  prime_numbers: set[int] = {2, 3, 5, 7}
  screen_resolution: tuple[int, int] = (1920, 1080)
  movie_ratings: list[float | str] = [1.5, "Great movie!", "N/A", 3.5]
  username: str | None = get_username(user_id) if user_is_logged_in(user_id) else None
  ```
  ```python
  # CORRECT - Python < 3.10
  from typing import Dict, List, Optional, Set, Tuple, Union

  name: str = "Amadeus"
  age: int = 66
  success_rate: float = 0.95
  machine_has_stopped: bool = False
  payload: bytes = b'Sensor_Data'
  last_name_to_age: Dict[str, int] = {"Mozart": 66, "Beethoven": 99}
  grades: List[int] = [1]
  prime_numbers: Set[int] = {6, 7}
  screen_resolution: Tuple[int, int] = (1920, 1080)
  movie_ratings: list[Union[float, str]] = [1.5, "Great movie!", "N/A", 3.5]
  username: Optional[str] = get_username(user_id) if user_is_logged_in (user_id) else None
  ```
- Declare return_type
  - return_type=None if function has no return statement. 
  ```python 
  def log_error(error_message: str) -> None
    print(f"[ERROR] {error_message}")
  ```
  - else define correct type
  ```python 
  # Python 3.10+
  def find_maximum(values: list[int]) -> int | None:
      if values:
          return max(values)
      else:
          return None
  ```
  ```python 
  # Python <3.10
  from typing import List, Optional

  def find_maximum(values: List[int]) -> Optional[int]:
      if values:
          return max(values)
      else:
          return None
  ```

- Use inline comment "# type ignore" when importing third_party modules, that don't use type hinting

##  6. Import Modules Correctly
  - Imports should be grouped into `Standard library imports`/`Third party imports`/`Local imports`, seperated with a blank line inbetween and use stand-alone comments above each category section. (if a category is not used, then the comment should be deleted)
    ```python
    # WRONG
    import requests
    from local_module import local_component
    import time
    from flask import Flask
    import os
    ```    
    ```python
    # CORRECT
    # Standard library imports
    import time
    import os

    # Third party imports
    from flask import Flask
    import requests

    # Local imports
    from local_module import local_component
    ```

  - Imports should be at the top of the file and never in a function to avoid performance losses.
    ```python
    # WRONG
    import module

    def function_name(...) -> None:
        import another_module
        ...
    ```
    ```python
    # CORRECT
    import module
    import another_module

    def function_name(...) -> None:
        ...
    ```

  - All imports should be sorted alphabetically and on seperate lines (top to bottom).
        ```python
        # WRONG
        import module, another_module
        ```
        ```python
        # CORRECT
        import another_module
        import module
        ``` 
  - When importing more than 3 components from the same module, use `import module`. (Exception is module `typing`) 
        ```python
        # CORRECT
        import module
        ```
        ```python
        # WRONG
        from module import component_1, component_2, component_3, component_4 
        ```

  - When importing less than 4 components from the same module, use `from module import component`.
        ```python
        # CORRECT
        from module import component_1, component_2, component_3 
        ```
        ```python
        # WRONG
        import module
        module.component_1()
        module.component_2()
        module.component_3()
        ```
        
  - All imported components should be sorted alphabetically (left to right).
        ```python
        # WRONG
        from module 
        ```
        ```python
        # CORRECT
        import another_module
        import module
        ```

  - If necessary, use `import module as alias` to simplify long module names. 
        ```python
        # CORRECT
        import very_very_long_module_name as alias
        ```
        ```python
        # WRONG
        import very_very_long_module_name
        ```

  - Sometimes established Aliases can be used for commonly used modules.
        ```python
        # CORRECT
        import matplotlib.pyplot as plt
        import pandas as pd
        ```

  - Sometimes Aliases are necessary if modules use the same naming on classes or functions.
        ```python
        # CORRECT
        from module_1 import component as component_1
        from module_2 import component as component_2
        ```
        ```python
        # WRONG
        from module_1 import component 
        from module_2 import component 
        ```
  - Avoid wildcards imports `from module import *` as it makes the source of components unclear.
        ```python
        # CORRECT
        from module import component 
        ```
        ```python
        # WRONG
        from module import *
        ```
  - Imports should always be absolute and not relative
        ```python
        # CORRECT
        from package_1.module_1 import component_1
        ```
        ```python
        # WRONG
        from ..package_1.module_1 import component_1
        ```

  - Remove imports, that are not used in the file
        ```python
        # CORRECT
        from module_2 import function_2
        
        function_2()
        ```
        ```python
        # WRONG
        from module_1 import function_1
        from module_2 import Class_2, function_2
        
        function_2()
        ```

##  7. Docstrings
DOCSTRINGS must be done for CLASSES, FUNCTIONS/METHODS and MODULES!
Do not remove any comments! (but still do the module docstring)
### General Docstring rules
  - No blank line before the Docstring
    ```python
    # WRONG
    def function_name(...) -> return_type:

      """
      Docstring goes here.
      """
      return return_value
    ```
    ```python
    # CORRECT
    def function_name(...) -> return_type:
      """
      Docstring goes here.
      """
      return return_value
    ```

  - The indentation of the docstring should match the given code.
    ```python
    # WRONG
    def function_name(...) -> return_type:
      """Docstring goes here.
      """

      return return_value
    ```
    ```python
    # WRONG
    def function_name(...) -> return_type:
      """
      Docstring goes here."""

      return return_value
    ```
    ```python
    # CORRECT
    def function_name(...) -> return_type:
      """
      Docstring goes here.
      """
      return return_value
    ```

  - The docstring must be a complete sentence (ending with a period), describe the function's behavior as a command ("Do X & Return Y") and avoid passive descriptions ("Returns X").
    ```python
    # WRONG
    def add_numbers(a: int, b: int) -> int:
        """
        Returns sum of two numbers
        """
        return a + b
    ```
    ```python
    # CORRECT
    def add_numbers(a: int, b: int) -> int:
        """
        Add two numbers and return their sum.
        """
        return a + b
    ```

  - The docstring should not contain redundant repetitions.
    ```python
    # WRONG
    def add_numbers(a: int, b: int) -> int:
        """
        add_numbers(a, b) -> int
        """
        return a + b
    ```
    ```python
    # CORRECT
    def add_numbers(a: int, b: int) -> int:
        """
        Add two numbers and return their sum.
        """
        return a + b
    ```

  - If the docstring contains a backslash (\\), use r"""raw triple quotes""" to avoid formatting issues.
    ```python
    # WRONG
    def read_file(file_path: str) -> str:
        """
        ...

        Examples
        --------
        >>> read_file(r"C:\Users\Username\new_folder\file.txt")
        Hello there!
        """
        with open(file_path, 'r') as file:
            return file.read()    
    ```
    ```python
    # CORRECT
    def read_file(file_path: str) -> str:
        r"""
        ...
        
        Examples
        --------
        >>> read_file(r"C:\Users\Username\new_folder\file.txt")
        Hello there!
        """
        with open(file_path, 'r') as file:
            return file.read()
    ```

### Add Function/Method Docstrings by using the following rules
  - No blank line after the function/method docstring
    ```python
    # WRONG
    def function_name(...) -> return_type:
      """
      Docstring goes here.
      """

      return return_value
    ```
    ```python
    # CORRECT
    def function_name(...) -> return_type:
      """
      Docstring goes here.
      """
      return return_value
    ```

  - For methods: Self must not be listed in the parameter section
    ```python
    # WRONG
    class Calculator:
        def add(self, a: int, b: int) -> int:
            """
            ...

            Parameters
            ----------
            self : Calculator
                The instance of the class.
            a : int
                First number.
            b : int
                Second number.

            ...
            """
            return a + b
    ```
    ```python
    # CORRECT
    class Calculator:
        def add(self, a: int, b: int) -> int:
            """
            ...

            Parameters
            ----------
            a : int
                First number.
            b : int
                Second number.

            ...
            """
            return a + b
    ```

  - Function & method docstrings should always follow this pattern:
    ```python
    def function_name(param1: param1_type, param2: param2_type) -> return_type:
            """
            Brief description of the function as an imperative command ("Do X & return Y") 

            Detailed explanation of what the function does. 
            Why is it needed? Are Error raised? If so, when?
            What steps or calculations does it perform?

            Parameters
            ----------
            param1: param1_type
                Description of the first parameter.
            param2: param2_type
                Description of the second parameter.

            Returns
            -------
            return_type
                Description of what the function returns.

            Raises (if applicable)
            -------
            <ErrorType>
                Description of the exception raised if an error occurs.

            Examples
            --------
            >>> function_name(param1_value, param2_value)
            <expected_return_value_of_given_example>
            """
            return return_value
    ```

  - This is an example, where the given pattern has been applied.
    ```python
    def calculate_total_with_tax(prices: list[int | float], tax_rate: float) -> float:
        """
        Sums up given prices, applies the given tax rate, and returns the total price.

        This function sums up the provided price list and applies the given tax rate.
        It raises a ValueError if the tax rate is negative or if any price is
        either non-numeric or negative.

        Parameters
        ----------
        prices: list[int | float]
            A list of item prices.
        tax_rate: float
            The tax rate to apply.

        Returns
        -------
        float
            The total price after applying the tax.

        Raises
        ------
        ValueError
            If the tax rate is negative, or
            if any price is non-numeric or negative.

        Examples
        --------
        >>> calculate_total_with_tax([100.2, 50, 25.5], 0.1)
        193.27
        """
        if tax_rate < 0:
            raise ValueError(f"Invalid tax rate: {tax_rate}. Must be a non-negative value.")

        prices_are_non_numeric = not all(
            isinstance(price, (int, float)) for price in prices
        )
        if prices_are_non_numeric:
            raise ValueError("All prices must be of numeric (int/float)")

        prices_are_negative = any(price < 0 for price in prices)
        if prices_are_negative:
            raise ValueError("All prices must be non-negative.")

        total: float = sum(prices)
        total_with_tax = total * (1 + tax_rate)
        return total_with_tax
    ```

### Add class docstrings with the following rules
   - One blank line after the class docstring
    ```python
    # WRONG
    class SomeClass:
        """
        ...
        """
        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1
    ```
    ```python
    # CORRECT
    class SomeClass:
        """
        ...
        """

        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1
    ```

  - The Methods section of a class-docstring should not list `self` as the first parameter, nor should it include any private methods.
    ```python
    # WRONG
    class SomeClass:
        """
        ...

        Methods
        -------
        method1(param1)
            Brief description of what method1 does.
        _private_method()
            This is a private method and should not be listed.

        ...
        """

        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1

        def _private_method(self) -> None:
            pass

        ...
    ```
    ```python
    # CORRECT
    class SomeClass:
        """
        ...

        Methods
        -------
        method1(param1)
            Brief description of what method1 does.

        ...
        """

        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1
            
        def _private_method(self) -> None:
            pass

        ...
    ```

  - Always document all attributes in class-docstrings, this includes those initialized via the constructor and those initialized internally with a default value.
    ```python
    # WRONG
    class SomeClass:
        """
        ...

        Attributes
        ----------
        attribute1: attribute1_type
            Description of the first attribute (initialized via the constructor).

        ...
        """

        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1
            self.attribute2: attribute1_type = False

        ...
    ```
    ```python
    # CORRECT
    class SomeClass:
        """
        ...

        Attributes
        ----------
        attribute1: attribute1_type
            Description of the first attribute (initialized via the constructor).
        attribute2: attribute2_type
            Description of the second attribute (initialized internally).
        ...
        """

        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1
            self.attribute2: attribute1_type = False

        ...
    ```

  - Class docstrings should always follow this pattern:
    ```python 
    class SomeClass:
        """
        Brief description of the class.

        Detailed explanation of what the class does.
        Why is it needed? Are Error raised? If so, when?
        What steps or calculations does it perform?

        Attributes
        ----------
        attribute1: attribute1_type
            Description of the first attribute.
        attribute2: attribute2_type
            Description of the second attribute.

        Methods
        -------
        method1(param1)
            Description of what method1 does.
        """

        def __init__(self, attribute1: attribute1_type) -> None:
            self.attribute1: attribute1_type = attribute1
            self.attribute2: attribute1_type = False

        def method1(self, param1: param1_type) -> return_type:
            ...

        def _private_method(self) -> None:
            ...
    ```
  - This is an example, where the given pattern has been applied.
    ```python
    class Machine:
        """
        A class used to represent a Machine.

        This class models a basic machine with functionality to start and stop. 
        It keeps track of its running state.

        Attributes
        ----------
        brand: str
            The brand of the machine.
        model: str
            The model of the machine.
        is_running: bool
            Indicates whether the machine is currently running.

        Methods
        -------
        start():
            Starts the machine, changing its state to running.
        stop():
            Stops the machine, changing its state to not running.
        """

        def __init__(self, brand: str, model: str) -> None:
            """
            Constructs all the necessary attributes for the Machine object.

            Parameters
            ----------
            brand: str
                The brand of the machine.
            model: str
                The model of the machine.
            """
            self.brand: str = brand
            self.model: str = model
            self.is_running: bool = False

        def start(self) -> None:
            """
            Starts the machine.

            This method changes the state of the machine to running and outputs 
            a message indicating that the machine has started.
            """
            self.is_running = True
            print(f"{self.brand} {self.model} started.")

        def stop(self) -> None:
            """
            Stops the machine.

            This method changes the state of the machine to not running and 
            outputs a message indicating that the machine has stopped.
            """
            self.is_running = False
            print(f"{self.brand} {self.model} stopped.")
    ```

### Add module docstrings with the following rules
   - Two blank lines after the module docstring
    ```python
    # WRONG
    """
    Module name
    ===========
    ...
    """
    class SomeClass:
        ...
    ```
    ```python
    # WRONG
    """
    Module name
    ===========
    ...
    """

    class SomeClass:
        ...
    ```
    ```python
    # CORRECT
    """
    Module name
    ===========
    ...
    """


    class SomeClass:
        ...
    ```

  - Module docstrings should always follow this pattern:
    ```python
    """
    Module name
    ===========

    Brief description of the module.

    Detailed explanation of the module (including its purpose and any other relevant details)

    Examples
    --------
    Provide example usage patterns.

    >>> from module_name import some_function
    >>> result = some_function(arguments)
    >>> print(result)

    >>> from module_name import SomeClass
    >>> some_object = SomeClass(arguments)
    >>> some_object.some_method()
    """


    def some_function():
        ...


    class SomeClass:
        ...
    
    ```

  - This is an example, where the given pattern has been applied.
    ```python
    """    
    Machine utils
    =============

    A simple module for representing and managing machines.

    This module provides a class to model a basic machine with the ability to start and stop. 
    It keeps track of the machine's running state and provides methods to control it.

    Examples
    --------
    >>> from machine_utils import Machine
    >>> machine_1 = Machine("Honda", "GX200")
    >>> machine_1.start()
    Honda GX200 started.

    >>> machine_1.stop()
    Honda GX200 stopped.
    """


    class Machine:
        """
        ...
        """

        def __init__(self, brand: str, model: str) -> None:
            """
            ...
            """
            self.brand: str = brand
            self.model: str = model
            self.is_running: bool = False

        def start(self) -> None:
            """
            ...
            """
            self.is_running = True
            print(f"{self.brand} {self.model} started.")

        def stop(self) -> None:
            """
            ...
            """
            self.is_running = False
            print(f"{self.brand} {self.model} stopped.")
    ```

##  8. Formatting & spacing

### Make sure unused imports are removed

### For strings use double quote strings ("), not single quotes(')
  - Be careful, when changing f-strings to not impact the codes funcationality

### Make the code PEP-8 compliant

### Limit the line length
  - Limit long blocks of text (docstrings or comments) to maximum 72 characters.
  - Limit all other lines to maximum 79 characters.
  - Be careful to not accidentally remove any necessary characters (like f-strings)