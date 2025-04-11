"""
OptiPy
======

Optipy is a code optimization tool that uses various LLM providers to improve Python code quality.

This module provides functionality to optimize Python code according to specified guidelines
using different LLM providers (Anthropic, Google, OpenAI) and optimization strategies.

Examples
--------
**Command-line usage:**
.. code-block:: bash
$ python optipy.py --filepath=example.py --strategy=one_by_one --model=gpt-4o --debug

**Library usage:**
.. code-block:: python
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
"""

# Standard library imports
import argparse
import os
from pathlib import Path
import re
from typing import get_args, Literal, TypeAlias, TypedDict
import time

# Third party imports
import anthropic
import autopep8  # type: ignore
from dotenv import load_dotenv
from google import genai  # type: ignore
from google.genai import types  # type: ignore
from openai import OpenAI

# Local imports
import toolbox


OptimizationStrategy: TypeAlias = Literal["all_at_once", "one_by_one"]


class LLMResponse(TypedDict):
    """
    A TypedDict representing a response from a Language Learning Model (LLM).

    This structure encapsulates the content returned by the LLM along with token usage metrics
    for both input and output. It provides a standardized format for responses
    across different LLM providers.

    Attributes
    ----------
    content : str
        The text content returned by the LLM.
    input_tokens : int
        The number of tokens used in the input prompt.
    output_tokens : int
        The number of tokens generated in the output response.
    """

    content: str
    input_tokens: int
    output_tokens: int


class OptipyConfig:
    """
    Configuration class for Optipy to manage and validate optimization settings.

    This class handles the selection of the LLM provider, model, API key retrieval,
    and provides a structured way to access available models. It ensures the chosen
    model belongs to a recognized provider and retrieves the appropriate API key
    from environment variables.

    Parameters
    ----------
    filepath : Path
        The path to the Python file to be optimized.
    strategy : OptimizationStrategy
        The strategy for optimization ('one_by_one' or 'all_at_once').
    model : str | None, optional
        The specific LLM model to use. Defaults to OpenAI's "gpt-4o-2024-08-06" if None is provided.
    debug : bool, optional
        Whether to enable debug mode for logging outputs, by default False.

    Attributes
    ----------
    filepath : Path
        Stores the file path of the Python script to be optimized.
    strategy : OptimizationStrategy
        Optimization strategy to use.
    model : str
        Selected model for the optimization.
    provider : str
        The LLM provider associated with the selected model (Anthropic, Google, OpenAI, or XAI).
    api_key : str
        API key retrieved from environment variables for the selected provider.

    Methods
    -------
    get_api_key() -> str
        Returns the API key for the configured provider.
    get_available_models() -> dict[str, list[str]]
        Returns a dictionary of all available models grouped by provider.
    """

    PROVIDERS = {
        "anthropic": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ],
        "google": [
            "chat-bison-001",
            "text-bison-001",
            "embedding-gecko-001",
            "gemini-1.0-pro-vision-latest",
            "gemini-pro-vision",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-001",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-001-tuning",
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash-8b-001",
            "gemini-1.5-flash-8b-latest",
            "gemini-1.5-flash-8b-exp-0827",
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-lite-preview-02-05",
            "gemini-2.0-flash-lite-preview",
            "gemini-2.0-pro-exp",
            "gemini-2.0-pro-exp-02-05",
            "gemini-exp-1206",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.0-flash-thinking-exp-1219",
            "learnlm-1.5-pro-experimental",
            "embedding-001",
            "text-embedding-004",
            "gemini-embedding-exp-03-07",
            "gemini-embedding-exp",
            "aqa",
            "imagen-3.0-generate-002",
        ],
        "openai": [
            "gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27",
            "gpt-4o-mini-audio-preview-2024-12-17",
            "dall-e-3",
            "dall-e-2",
            "gpt-4o-audio-preview-2024-10-01",
            "gpt-4o-audio-preview",
            "gpt-4o-mini-realtime-preview-2024-12-17",
            "gpt-4o-mini-realtime-preview",
            "o1-mini-2024-09-12",
            "o1-mini",
            "omni-moderation-latest",
            "gpt-4o-mini-audio-preview",
            "omni-moderation-2024-09-26",
            "whisper-1",
            "gpt-4o-realtime-preview-2024-10-01",
            "babbage-002",
            "gpt-4-turbo-preview",
            "chatgpt-4o-latest",
            "tts-1-hd-1106",
            "text-embedding-3-large",
            "gpt-4-0125-preview",
            "gpt-4o-audio-preview-2024-12-17",
            "gpt-4",
            "gpt-4o-2024-05-13",
            "tts-1-hd",
            "o1-preview",
            "o1-preview-2024-09-12",
            "gpt-4o-2024-11-20",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",
            "tts-1",
            "tts-1-1106",
            "davinci-002",
            "gpt-3.5-turbo-1106",
            "gpt-4-turbo",
            "gpt-3.5-turbo-instruct",
            "o1",
            "gpt-4o-2024-08-06",
            "gpt-3.5-turbo-0125",
            "gpt-4o-realtime-preview-2024-12-17",
            "gpt-3.5-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-realtime-preview",
            "gpt-3.5-turbo-16k",
            "gpt-4o",
            "text-embedding-3-small",
            "gpt-4-1106-preview",
            "text-embedding-ada-002",
            "o3-mini-2025-01-31",
            "gpt-4-0613",
            "o3-mini",
            "o1-2024-12-17",
        ],
        "xai": [
            "grok-2-1212",
            "grok-2-vision-1212",
            "grok-beta",
            "grok-vision-beta",
        ],
    }

    DEFAULT_MODELS = {
        "anthropic": "claude-3-7-sonnet-20250219",
        "google": "gemini-2.0-pro-exp-02-05",
        "openai": "gpt-4o-2024-08-06",
        "xai": "grok-2-1212",
    }

    def __init__(
        self,
        filepath: Path,
        strategy: OptimizationStrategy,
        model: str | None = None,
        debug: bool = False,
    ) -> None:
        """Initializes the configuration for Optipy."""
        self.filepath = filepath
        self.strategy = strategy
        self.model = model or self.DEFAULT_MODELS["openai"]
        self.debug = debug
        self.provider = self._get_provider()
        self.api_key = self._read_env()

    def _get_provider(self) -> str:
        """Determines the provider based on the selected model."""
        for provider, models in self.PROVIDERS.items():
            if self.model in models:
                return provider
        available_models = [
            model for models in self.PROVIDERS.values() for model in models
        ]
        raise ValueError(
            f"Unknown model: {self.model}. Available models: {', '.join(sorted(available_models))}"
        )

    def _read_env(self) -> str:
        """Loads and retrieves the API key from the environment."""
        load_dotenv(override=True)
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
        }
        api_key = os.getenv(env_vars[self.provider])
        if api_key is None:
            raise ValueError(f"{env_vars[self.provider]} environment variable not set")
        return api_key

    def get_api_key(self) -> str:
        """Returns the API key for the configured provider."""
        return self.api_key

    def get_available_models(self) -> dict[str, list[str]]:
        """Returns a dictionary of all available models grouped by provider."""
        return self.PROVIDERS


class Optipy:
    """
    A class to optimize Python code based on predefined guidelines using different LLM providers.

    Parameters
    ----------
    filepath : Path
        The path to the Python code file that needs optimization.
    strategy : OptimizationStrategy
        The optimization strategy to use ('one_by_one' or 'all_at_once').
    model : str | None, optional
        The LLM model to use for optimization, by default None.
    debug : bool, optional
        Whether to enable debug mode for logging outputs, by default False.

    Attributes
    ----------
    config : OptipyConfig
        Configuration settings for the optimization process.
    guideline : str
        The content of the guideline file used for optimization.
    guideline_rules : list[str]
        The extracted rules from the guideline.
    guideline_titles : list[str]
        The extracted titles of the guideline rules.
    original_code : str
        The original Python code before optimization.
    optimized_code : str | None
        The optimized Python code after processing, or None if not optimized yet.

    Methods
    -------
    run_optimization()
        Runs the optimization process using the selected strategy.
    get_original_code() -> str
        Returns the original Python code before optimization.
    get_optimized_code() -> str
        Returns the optimized Python code after processing.
    """

    def __init__(
        self,
        filepath: Path,
        strategy: OptimizationStrategy,
        model: str | None = None,
        debug: bool = False,
    ) -> None:
        """Initializes Optipy with configuration settings."""
        self.config = OptipyConfig(filepath, strategy, model, debug)
        self.guideline = self._read_guideline()
        self.guideline_rules = self._extract_guideline_rules()
        self.guideline_titles = self._extract_guideline_titles()
        self.original_code = self._read_input_code()
        self.optimized_code: str | None = None

    def _get_llm_response(self, original_code: str, prompt: str) -> LLMResponse:
        """Fetches response from the selected LLM provider."""
        if self.config.provider == "anthropic":
            return self._get_anthropic_response(original_code, prompt)
        if self.config.provider == "google":
            return self._get_google_response(original_code, prompt)
        if self.config.provider == "openai":
            return self._get_openai_response(original_code, prompt)
        if self.config.provider == "xai":
            return self._get_xai_response(original_code, prompt)
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _get_anthropic_response(self, original_code: str, prompt: str) -> LLMResponse:
        """Gets a response from Anthropic provider."""
        client = anthropic.Anthropic(api_key=self.config.api_key)
        message = client.messages.create(
            model=self.config.model,
            temperature=0,
            max_tokens=8192,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": original_code}],
                }
            ],
        )

        if not hasattr(message.content[0], "text"):
            raise ValueError("Invalid response from Anthropic API")

        response: LLMResponse = {
            "content": message.content[0].text,
            "input_tokens": int(message.usage.input_tokens),
            "output_tokens": int(message.usage.output_tokens),
        }
        return response

    def _get_google_response(self, original_code: str, prompt: str) -> LLMResponse:
        """Gets a response from Google provider."""
        client = genai.Client(api_key=self.config.api_key)
        response = client.models.generate_content(
            model=self.config.model,
            contents=original_code,
            config=types.GenerateContentConfig(
                system_instruction=prompt, temperature=0
            ),
        )
        response_data: LLMResponse = {
            "content": response.text,
            "input_tokens": int(response.usage_metadata.prompt_token_count),
            "output_tokens": int(response.usage_metadata.candidates_token_count),
        }
        return response_data

    def _get_openai_response(self, original_code: str, prompt: str) -> LLMResponse:
        """Gets a response from OpenAI provider."""
        client = OpenAI(api_key=self.config.api_key)
        chat_completion = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": original_code},
            ],
            temperature=0,
        )

        if not chat_completion.usage or not chat_completion.choices[0].message.content:
            raise ValueError("Invalid response from OpenAI API")

        response_data: LLMResponse = {
            "content": chat_completion.choices[0].message.content,
            "input_tokens": chat_completion.usage.prompt_tokens,
            "output_tokens": chat_completion.usage.completion_tokens,
        }
        return response_data

    def _get_xai_response(self, original_code: str, prompt: str) -> LLMResponse:
        """Gets a response from XAI provider."""
        client = OpenAI(
            api_key=self.config.api_key,
            base_url="https://api.x.ai/v1",
        )
        chat_completion = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": original_code},
            ],
            temperature=0,
        )

        if not chat_completion.usage or not chat_completion.choices[0].message.content:
            raise ValueError("Invalid response from XAI API")

        response_data: LLMResponse = {
            "content": chat_completion.choices[0].message.content,
            "input_tokens": chat_completion.usage.prompt_tokens,
            "output_tokens": chat_completion.usage.completion_tokens,
        }
        return response_data

    def _read_guideline(self) -> str:
        """Reads the guideline file for optimization rules."""
        with Path("guideline.md").open("r", encoding="utf-8") as f_in:
            return f_in.read()

    def _extract_guideline_rules(self) -> list[str]:
        """Extracts individual rules from the guideline."""
        return self.guideline.split("##  ")[1:]

    def _extract_guideline_titles(self) -> list[str]:
        """Extracts titles of guideline rules."""
        pattern: str = r"^##\s+(?:\d+\.\s*)?(.+)"
        return re.findall(pattern, self.guideline, re.MULTILINE)

    def _read_input_code(self) -> str:
        """Reads the input Python file for optimization."""
        with self.config.filepath.open("r", encoding="utf-8") as f_in:
            return f_in.read()

    def _create_optimization_prompt(self, rules: str | list[str]) -> str:
        """Creates a prompt for the LLM optimization request."""
        rules_text: str
        if isinstance(rules, list):
            joined_rules: str = "\n## ".join(rules)
            rules_text = f"rules: \n## {joined_rules}"
        else:
            rules_text = f"rule: \n## {rules}"

        optimization_prompt: str = f"""# Role
- You are an Expert in establishing code quality in Python (3.10+).

# Context
- Give back the code in markdown format.
- Do not include any explanation.
- Modify only what is mentioned in the provided guidelines.
- If the code is not compliant with the given guideline, you will be fined $1000! 

# Instructions 
- Your task is to ensure that the given code adheres to the following {rules_text}          
"""
        return optimization_prompt

    def _extract_optimized_code(self, llm_response: str) -> str:
        """Extracts the optimized code from the LLM response."""
        if not llm_response:
            raise ValueError("No response received from the LLM.")
        pattern: str = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, llm_response, re.DOTALL)
        if not matches:
            print(str(llm_response))
            raise ValueError(
                "No Python code block found in LLM response, check max_tokens/rate_limits!"
            )
        return matches[0]

    def _format_code(self, code: str) -> str:
        """Formats the code using autopep8."""
        return autopep8.fix_code(code)

    def _save_optimized_code(self, optimized_code: str) -> None:
        """Saves the optimized code to a new file."""
        suffix: str = (
            "_optimized_aao"
            if self.config.strategy == "all_at_once"
            else "_optimized_obo"
        )
        optimized_filepath = self.config.filepath.with_stem(
            self.config.filepath.stem + suffix
        )
        with optimized_filepath.open("w", encoding="utf-8") as f_out:
            f_out.write(optimized_code)

    def _optimize_one_by_one(self) -> str:
        """Optimizes code one rule at a time."""
        optimized_code = self.original_code
        for i, rule in enumerate(self.guideline_rules):
            start_time = time.time()
            prompt = self._create_optimization_prompt(rule)
            llm_response = self._get_llm_response(optimized_code, prompt)
            llm_content = llm_response["content"]
            used_input_tokens = llm_response["input_tokens"]
            used_output_tokens = llm_response["output_tokens"]
            optimized_code = self._extract_optimized_code(llm_content)
            execution_time = time.time() - start_time
            if self.config.model == "gemini-2.0-pro-exp-02-05" and execution_time < 12:
                time.sleep(12 - execution_time)
            execution_time = time.time() - start_time

            if self.config.debug:
                print(optimized_code)

            print(
                f"✅ [{i+1}/{len(self.guideline_titles)}]"
                f"[{self.guideline_titles[i]}] Successfully applied rule in "
                f"{execution_time:.2f}s "
                f"({used_input_tokens=} | {used_output_tokens=})"
            )

        return optimized_code

    def _optimize_all_at_once(self) -> str:
        """Optimizes code by applying all rules at once."""
        start_time = time.time()
        prompt = self._create_optimization_prompt(self.guideline_rules)
        llm_response = self._get_llm_response(self.original_code, prompt)
        llm_content = llm_response["content"]
        used_input_tokens = llm_response["input_tokens"]
        used_output_tokens = llm_response["output_tokens"]
        optimized_code = self._extract_optimized_code(llm_content)
        execution_time = time.time() - start_time

        if self.config.debug:
            print(optimized_code)

        print(
            f"✅ Applied all {len(self.guideline_titles)} rules at once in "
            f"{execution_time:.2f}s ({used_input_tokens=} | {used_output_tokens=})"
        )

        return optimized_code

    @toolbox.measure_exec_time
    def run_optimization(self) -> None:
        """Run the optimization process using the selected strategy."""
        if self.config.strategy == "one_by_one":
            optimized_code = self._optimize_one_by_one()
        else:
            optimized_code = self._optimize_all_at_once()

        formatted_code = self._format_code(optimized_code)
        if self.config.debug:
            print(formatted_code)
        self.optimized_code = formatted_code
        self._save_optimized_code(formatted_code)

    def get_original_code(self) -> str:
        """Returns the original code before optimization."""
        return self.original_code

    def get_optimized_code(self) -> str:
        """Returns the optimized code after processing."""
        if self.optimized_code is None:
            raise ValueError("No optimized code available. Run optimize() first.")
        return self.optimized_code


def get_optipy_args() -> argparse.Namespace:
    """Parse command line arguments for file path, strategy, model, and debug options."""
    parser = argparse.ArgumentParser(
        description="Optimize code quality from a given file."
    )
    parser.add_argument(
        "--filepath", type=Path, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=get_args(OptimizationStrategy),
        default="one_by_one",
        help="Optimization strategy to use (default: one_by_one)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (default: gpt-4o from OpenAI)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print optimized code",
    )
    return parser.parse_args()


def main() -> None:
    """Executes the main functionality of the script."""
    args = get_optipy_args()

    optipy = Optipy(
        filepath=args.filepath,
        strategy=args.strategy,
        model=args.model,
        debug=args.debug,
    )
    optipy.run_optimization()
    # print(optipy.get_original_code())
    # print(optipy.get_optimized_code())


if __name__ == "__main__":
    main()
