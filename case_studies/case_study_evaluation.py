# Standard library imports
from pathlib import Path
import time
from typing import List, Literal

# Third party imports
from metrics_analyzer import MetricsAnalyzer
from optipy import Optipy


def run_optimization(
    model: str, filepaths: List[Path], strategy: Literal["all_at_once", "one_by_one"]
) -> None:
    """Run optimization for multiple files using specified strategy.

    Args:
        model: The model identifier to use for optimization
        filepaths: List of files to optimize
        strategy: Optimization strategy ("all_at_once" or "one_by_one")
    """
    if strategy == "all_at_once":
        strategy_suffix = "aao"
    else:
        strategy_suffix = "obo"

    for filepath in filepaths:
        optipy = Optipy(
            filepath=filepath,
            strategy=strategy,
            model=model,
            debug=False,
        )
        optipy.run_optimization()

        new_filepath = filepath.with_stem(
            f"{filepath.stem}_optimized_{strategy_suffix}"
        )
        analyzer = MetricsAnalyzer(new_filepath)
        analyzer.get_metrics()
        analyzer.display_metrics()

        time.sleep(10)  # Sleep between operations to prevent rate limiting


def main() -> None:
    models = [
        "claude-3-7-sonnet-20250219",
        "gemini-2.0-pro-exp-02-05",
        "gpt-4o-2024-08-06",
        "grok-2-1212",
    ]

    case_studies = [Path(f"./case_studies/case_study_1_{i}.py") for i in range(1, 6)]

    for model in models:
        run_optimization(model=model, filepaths=case_studies, strategy="all_at_once")
        run_optimization(model=model, filepaths=case_studies, strategy="one_by_one")


if __name__ == "__main__":
    main()
