"""
Metrics Analyzer
================

Module for evaluating and analyzing Python file metrics including
complexity and maintainability.

This module provides tools to analyze Python code for various quality
metrics.

Examples
--------
**Command-line usage:**
.. code-block:: bash
$ python metrics_analyzer.py --filepath=example.py

**Library usage:**
.. code-block:: python
    from metrics_analyzer import MetricsAnalyzer
    analyzer = MetricsAnalyzer("example.py")
    metrics = analyzer.get_metrics()
    analyzer.display_metrics()
"""

# Standard library imports
import argparse
import ast
import math
import re
import subprocess

# Third party imports
import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze


class MetricsAnalyzer:
    """Class for analyzing Python file metrics including complexity and maintainability."""

    def __init__(self, file_path) -> None:
        """Initialize the analyzer with a file path.

        Args:
            file_path: Path to the Python file to analyze
        """
        self.file_path = file_path
        self.code = None
        self.tree = None
        self.metrics = None

    def read_file(self) -> bool:
        """Read the file content."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as file_handle:
                self.code = file_handle.read()
                return True
        except (IOError, FileNotFoundError, PermissionError) as exc:
            print(f"Error reading file: {exc}")
            return False

    def get_ast_tree(self) -> bool:
        """Parse the file and return the AST."""
        if not self.code:
            return False
        try:
            self.tree = ast.parse(self.code, filename=self.file_path)
            return True
        except SyntaxError:
            print(f"Syntax error in {self.file_path}")
            return False

    def get_function_length(self) -> float | int:
        """Calculate the average function length."""
        if not self.tree:
            return 0

        function_lengths = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = max(n.lineno for n in ast.walk(node) if hasattr(n, "lineno"))
                function_lengths.append(end_line - start_line + 1)

        return sum(function_lengths) / len(function_lengths) if function_lengths else 0

    def get_function_length_without_docstrings(self) -> float | int:
        """Calculate the average function length without counting docstring lines."""
        if not self.tree:
            return 0

        function_lengths = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = max(n.lineno for n in ast.walk(node) if hasattr(n, "lineno"))
                total_length = end_line - start_line + 1

                docstring = ast.get_docstring(node)
                docstring_lines = 0

                if (
                    docstring
                    and node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Str)
                ):
                    docstring_node = node.body[0]

                    if hasattr(docstring_node, "end_lineno"):
                        docstring_lines = (
                            docstring_node.end_lineno - docstring_node.lineno + 1
                        )
                    else:
                        docstring_text = docstring
                        docstring_lines = (
                            docstring_text.count("\n") + 2
                            if "\n" in docstring_text
                            else 1
                        )

                adjusted_length = total_length - docstring_lines
                function_lengths.append(adjusted_length)

        return sum(function_lengths) / len(function_lengths) if function_lengths else 0

    def get_docstring_ratio(self) -> float | int:
        """Calculate the percentage of documented constructs."""
        if not self.tree:
            return 0

        total_constructs = 0
        documented_constructs = 0
        for node in ast.walk(self.tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                total_constructs += 1
                if ast.get_docstring(node):
                    documented_constructs += 1

        return (
            (documented_constructs / total_constructs) * 100
            if total_constructs > 0
            else 0
        )

    def get_maintainability_index(self, halstead_volume, avg_cc, sloc, docstring_ratio):
        """Calculate the maintainability index."""
        mi_formula = (
            171
            - 5.2 * math.log(halstead_volume)
            - (0.23 * avg_cc)
            - (16.2 * math.log(sloc))
            + (50 * (math.sin(math.sqrt(2.4 * math.radians(docstring_ratio)))))
        )
        maintainability_index = 100 * mi_formula / 171
        return max(0, min(100, maintainability_index))

    def rank_maintainability_index(self, mi_score) -> str:
        """Assign a ranking letter based on the Maintainability Index (MI) score."""
        if mi_score >= 20:
            return "A"  # Very High
        if 10 <= mi_score < 20:
            return "B"  # Medium
        return "C"  # Extremely Low

    def rank_cyclomatic_complexity(self, cc_score) -> str:
        """Assign a ranking letter based on the Cyclomatic Complexity (CC) score."""
        if cc_score <= 5:
            return "A"  # Low - Simple block
        if 6 <= cc_score <= 10:
            return "B"  # Low - Well structured and stable block
        if 11 <= cc_score <= 20:
            return "C"  # Moderate - Slightly complex block
        if 21 <= cc_score <= 30:
            return "D"  # More than moderate - More complex block
        if 31 <= cc_score <= 40:
            return "E"  # High - Complex block, alarming
        return "F"  # Very high - Unstable block, error-prone

    def get_pylint_score(self):
        """Run pylint and get the score."""
        try:
            result = subprocess.run(
                ["pylint", "--disable=W1203,R1705", self.file_path],
                capture_output=True,
                text=True,
                check=False,
            )
            match = re.search(r"Your code has been rated at ([\d.]+)/10", result.stdout)
            return float(match.group(1)) if match else None
        except (subprocess.SubprocessError, ValueError, TypeError) as exc:
            print(f"Error running pylint: {exc}")
            return None

    def _extract_code_metrics(self):
        """Extract all code metrics from the given code and AST tree."""
        raw_metrics = analyze(self.code)

        complexity_results = cc_visit(self.code)
        halstead_results = h_visit(self.code)

        if complexity_results:
            total_complexity = sum(block.complexity for block in complexity_results)
            avg_cc = total_complexity / len(complexity_results)
        else:
            avg_cc = 0

        halstead_volume = halstead_results.total.volume

        avg_func_len = self.get_function_length()
        avg_func_len_no_docs = self.get_function_length_without_docstrings()
        docstring_ratio = self.get_docstring_ratio()
        pylint_score = self.get_pylint_score()
        maint_index = self.get_maintainability_index(
            halstead_volume, avg_cc, raw_metrics.sloc, docstring_ratio
        )

        return {
            "file_path": self.file_path,
            "loc": raw_metrics.loc,
            "sloc": raw_metrics.sloc,
            "single_comments": raw_metrics.single_comments,
            "multi": raw_metrics.multi,
            "blank_lines": raw_metrics.blank,
            "comments": raw_metrics.comments,
            "avg_func_len": avg_func_len,
            "avg_func_len_no_docs": avg_func_len_no_docs,
            "avg_cc": avg_cc,
            "mi": maint_index,
            "pylint_score": pylint_score,
        }

    def get_metrics(self):
        """Analyze the file and return all metrics."""
        if not self.read_file():
            print(f"Cannot analyze empty file: {self.file_path}")
            return None

        if not self.get_ast_tree():
            print(f"Cannot parse file: {self.file_path}")
            return None

        self.metrics = self._extract_code_metrics()
        return self.metrics

    def _calculate_percentage(self, value, total):
        """Calculate percentage of a value against a total."""
        return (value / total) * 100 if total > 0 else 0

    def _prepare_metrics_for_display(self):
        """Prepare metrics data for display by calculating ratios and formatting values."""
        if not self.metrics:
            return {}

        metrics = self.metrics
        loc = metrics["loc"]

        ratios = {
            "sloc": self._calculate_percentage(metrics["sloc"], loc),
            "single_comments": self._calculate_percentage(
                metrics["single_comments"], loc
            ),
            "multi": self._calculate_percentage(metrics["multi"], loc),
            "blank_lines": self._calculate_percentage(metrics["blank_lines"], loc),
            "comments": self._calculate_percentage(metrics["comments"], loc),
        }

        total_percentage = round(
            ratios["sloc"]
            + ratios["single_comments"]
            + ratios["multi"]
            + ratios["blank_lines"],
            2,
        )

        rounded = {
            "func_len": round(metrics["avg_func_len"], 2),
            "func_len_no_docs": round(metrics["avg_func_len_no_docs"], 2),
            "cc": round(metrics["avg_cc"], 2),
            "mi": round(metrics["mi"], 2),
            "pylint": (
                round(metrics["pylint_score"], 2)
                if metrics["pylint_score"] is not None
                else None
            ),
        }

        mi_rank = self.rank_maintainability_index(rounded["mi"])
        cc_rank = self.rank_cyclomatic_complexity(rounded["cc"])

        return {
            "LOC": [f"{loc} ({total_percentage}%)"],
            "SLOC": [f"{metrics['sloc']} ({ratios['sloc']:.2f}%)"],
            "SLCOM": [
                f"{metrics['single_comments']} ({ratios['single_comments']:.2f}%)"
            ],
            "MULTI": [f"{metrics['multi']} ({ratios['multi']:.2f}%)"],
            "BLANK": [f"{metrics['blank_lines']} ({ratios['blank_lines']:.2f}%)"],
            "/": ["/"],
            "Commented Lines": [f"{metrics['comments']} ({ratios['comments']:.2f}%)"],
            "Avg. func len (with docs)": [f"{rounded['func_len']:.2f}"],
            "Avg. func len (without docs)": [f"{rounded['func_len_no_docs']:.2f}"],
            "Avg. CC": [f"{rounded['cc']:.2f} ({cc_rank})"],
            "MI": [f"{rounded['mi']:.2f} ({mi_rank})"],
            "Pylint Score": [
                f"{rounded['pylint']:.2f}" if rounded["pylint"] is not None else "N/A"
            ],
        }

    def display_metrics(self):
        """Format and display metrics in a table."""
        if not self.metrics:
            print("No metrics to display. Run analyze() first.")
            return None

        display_data = self._prepare_metrics_for_display()
        if not display_data:
            print("No metrics to display.")
            return None

        metrics_df = pd.DataFrame(display_data, index=[self.metrics["file_path"]])
        df_string = metrics_df.to_string(justify="center", col_space=12)
        separator = "=" * len(max(df_string.split("\n"), key=len))
        print(f"{separator}\n{df_string}\n{separator}")
        return df_string


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Python file metrics")
    parser.add_argument("--filepath", type=str, help="Path to Python file")
    return parser.parse_args()


def main():
    """Execute the main functionality of the script."""
    args = parse_arguments()
    analyzer = MetricsAnalyzer(args.filepath)
    analyzer.get_metrics()
    analyzer.display_metrics()


if __name__ == "__main__":
    main()
