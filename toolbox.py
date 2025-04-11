"""
Toolbox
=======

Toolbox is a simple Python utility that provides a decorator to measure
the execution time of functions. This helps in performance analysis by
displaying how long a function takes to run.

This module defines:
- `measure_exec_time`: A decorator that prints the execution time of a function.
- `test`: A sample function to demonstrate the usage of `measure_exec_time`.

Examples
--------
**Library usage:**
.. code-block:: python
    from measure_exec import measure_exec_time

    @measure_exec_time
    def example_function():
        for i in range(0, 100):
            print(i)

    example_function()
"""

# Standard library imports
from functools import wraps
import time
from typing import Any, Callable


def measure_exec_time(func: Callable) -> Callable:
    """
    A decorator to measure the execution time of a function.

    Parameters:
        func (Callable): The function whose execution time is to be measured.

    Returns:
        Callable: The wrapped function that includes time measurement.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f}s to run.")
        return result

    return wrapper


@measure_exec_time
def test() -> None:
    """Test function to demonstrate measure_exec_time decorator"""
    for i in range(0, 100):
        print(i)


if __name__ == "__main__":
    test()
