# OptiPy

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**OptiPy** is a code optimization tool developed as part of my Master's Thesis. It leverages leading LLM providers (e.g., OpenAI) to automatically refactor and improve Python code by applying a guideline consisting of 52 clean-code principles. These principles are based on Python Enhancement Proposals (PEPs) and best practices from real-world development experience. 

For further information see: [Master Thesis](https://epub.technikum-wien.at/search/quick?query=OptiPy)

## Table of Contents

- [Prerequisites](#Prerequisites)
- [Installation](#Installation)
- [Configuration](#Configuration)
- [Usage](#Usage)
  - [Library Integration](#Library-Integration)
  - [Command-Line](#Command-Line)
  - [CI/CD-Pipeline](#CI/CD-Pipeline)

## Prerequisites

- Python 3.10+
- pip


## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/phgas/optipy.git
    cd optipy
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
1. **API Key(s):**
   - Set the necessary API key(s) depending on the provider you want to use.
   - Example:
        ```bash
        export ANTHROPIC_API_KEY="your-api-key-here"
        export GOOGLE_API_KEY="your-api-key-here"
        export OPENAI_API_KEY="your-api-key-here"
        export XAI_API_KEY="your-api-key-here"
        ```

2. **Guideline:**
   - If you want you can change/compress/extend the default guideline to your needs.
   - Ensure the file is named `guideline.md` and it exists in your working directory.
    

## Usage

### Library Integration
```python
from pathlib import Path
from optipy import Optipy

optipy = Optipy(
    filepath=Path("example.py"),
    strategy="one_by_one",
    model="gpt-4o",
    debug=True
)

optipy.run_optimization()
# print(optipy.get_original_code())
# print(optipy.get_optimized_code())
```

### Command-Line

```bash
python optipy.py --filepath=example.py --strategy=one_by_one --model=gpt-4o --debug
```
| Argument     | Description                                          | Default                      |
|--------------|------------------------------------------------------|------------------------------|
| `--filepath` | Path to the Python file to optimize                  | ❗️ Required ❗️                 |
| `--strategy` | Optimization strategy: `one_by_one` or `all_at_once` | `one_by_one`                 |
| `--model`    | Model to use for optimization (see available models) | `gpt-4o-2024-08-06` (OpenAI) |
| `--debug`    | Enable debug mode to print intermediate outputs      | `False`                      |


### CI/CD-Pipeline
1. **Enable Optimization:**
   - Open `.github/workflows/optipy-workflow.yml`.
   - Set `DISABLE_OPTIMIZATION=false` (line 40).
2. **Setup files/directories to exclude:**
    - Change excluded patterns (line 47):
        ```bash
        EXCLUDE_PATTERNS=(
            "./file.py" 
            "./directory/*"
        )
        ```
3. **Setup Github Actions secrets:**
   - Go to your projects `Actions secrets` (https://github.com/YOUR_NAME/YOUR_PROJECT/settings/secrets/actions).
   - Click `New repository secret`.
   - Setup API key(s) (names are same as in [Configuration](#Configuration)).
   - Click `Add secret`.
   - Optional: If necessary, repeat for other providers.

