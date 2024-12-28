import torch
from data import get_dataset
from model import load_model_and_tokenizer
import logging
from tqdm import tqdm
import ast
import inspect
import contextlib
import io
import timeout_decorator
import json
from datetime import datetime
from typing import List, Optional, Union, Dict, Tuple, Any
import math
import statistics
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO)


@dataclass
class ErrorStats:
    """Statistics for different types of errors encountered during final test attempts."""

    syntax_errors: int = 0
    type_errors: int = 0
    assertion_errors: int = 0
    timeout_errors: int = 0
    runtime_errors: int = 0
    indentation_errors: int = 0
    total_samples: int = 0


class ErrorTracker:
    """Tracks errors from final test attempts across all problems in the dataset."""

    def __init__(self):
        self.problem_errors: Dict[int, ErrorStats] = defaultdict(ErrorStats)
        self.total_errors = ErrorStats()

    def add_error(self, problem_idx: int, error_type: str):
        """Record an error for a specific problem."""
        if error_type == "SyntaxError" or error_type == "InvalidSyntax":
            self.problem_errors[problem_idx].syntax_errors += 1
            self.total_errors.syntax_errors += 1
        elif error_type == "TypeError":
            self.problem_errors[problem_idx].type_errors += 1
            self.total_errors.type_errors += 1
        elif error_type == "AssertionError":
            self.problem_errors[problem_idx].assertion_errors += 1
            self.total_errors.assertion_errors += 1
        elif error_type == "TimeoutError":
            self.problem_errors[problem_idx].timeout_errors += 1
            self.total_errors.timeout_errors += 1
        elif error_type == "IndentationError":
            self.problem_errors[problem_idx].indentation_errors += 1
            self.total_errors.indentation_errors += 1
        else:
            self.problem_errors[problem_idx].runtime_errors += 1
            self.total_errors.runtime_errors += 1

    def increment_total(self, problem_idx: int):
        """Increment the total number of samples for a problem."""
        self.problem_errors[problem_idx].total_samples += 1
        self.total_errors.total_samples += 1

    def get_problem_stats(self, problem_idx: int) -> dict:
        """Get error statistics for a specific problem."""
        return asdict(self.problem_errors[problem_idx])

    def get_total_stats(self) -> dict:
        """Get overall error statistics."""
        return asdict(self.total_errors)


def extract_function(code_string, function_name):
    """Extract a function definition from a code string."""
    try:
        # Clean up the code string
        code_string = code_string.strip()

        # If the code starts with the function definition, return it directly
        if code_string.startswith(f"def {function_name}"):
            return code_string

        # Try parsing as AST
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            # Try adding a newline at the start in case indentation is wrong
            try:
                tree = ast.parse("\n" + code_string)
            except SyntaxError:
                return None

        # Find the function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the line numbers for the function
                start = node.lineno - 1
                end = node.end_lineno

                # Extract the function lines
                lines = code_string.split("\n")
                function_code = "\n".join(lines[start:end])
                return function_code

        return None

    except Exception as e:
        logging.error(f"Error extracting function: {e}")
        return None


@timeout_decorator.timeout(5)  # 5 second timeout for execution
def execute_test_case(func_obj, test_case, test_env):
    """Execute a single test case and return True if it passes."""
    try:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(test_case, test_env)
        return True
    except AssertionError as e:
        logging.error(f"Test assertion failed: {str(e)}")
        return False
    except TypeError as e:
        logging.error(f"Type error in implementation: {str(e)}")
        return False
    except timeout_decorator.TimeoutError:
        logging.error("Test execution timed out - likely infinite loop detected")
        return False
    except Exception as e:
        logging.error(f"Error executing test case: {type(e).__name__}: {str(e)}")
        return False


def evaluate_model(model, tokenizer, dataset, num_problems, n_samples, k):
    """
    Evaluate the model on the dataset with error tracking for final attempts.
    """
    results = []
    error_tracker = ErrorTracker()
    device = next(model.parameters()).device

    for idx in tqdm(range(num_problems)):
        item = dataset[idx]
        question = item["question"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]
        correct_samples = 0

        for sample_idx in range(n_samples):
            error_tracker.increment_total(idx)

            try:
                encoded_input = tokenizer(
                    question, return_tensors="pt", truncation=True
                )
                input_ids = encoded_input["input_ids"].to(device)
                attention_mask = encoded_input.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        num_beams=5,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        min_length=50,
                        no_repeat_ngram_size=2,
                        early_stopping=False,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logging.info(f"\nRaw generated code:\n{response}\n")

                # Try running tests on raw response first
                test_env = create_test_env()
                if try_run_tests(response, entry_point, test_code, test_env):
                    correct_samples += 1
                    logging.info("✓ Sample passed all tests on raw response")
                    continue

                # Extract function and try fixes
                generated_code = ""
                if "def " + entry_point in response:
                    start = response.find("def " + entry_point)
                    generated_code = response[start:]

                    # Try AST parsing
                    try:
                        tree = ast.parse(generated_code)
                        for node in ast.walk(tree):
                            if (
                                isinstance(node, ast.FunctionDef)
                                and node.name == entry_point
                            ):
                                end = node.end_lineno
                                generated_code = "\n".join(
                                    generated_code.split("\n")[:end]
                                )
                                break
                    except SyntaxError:
                        # Fallback: manual parsing
                        lines = generated_code.split("\n")
                        result = []
                        in_docstring = False
                        docstring_delim = 0

                        for line in lines:
                            stripped = line.strip()
                            if '"""' in line or "'''" in line:
                                docstring_delim += line.count('"""') + line.count("'''")
                                in_docstring = docstring_delim % 2 != 0
                            if (
                                not in_docstring
                                and stripped
                                and not (
                                    line[0].isspace()
                                    or stripped.startswith(
                                        (
                                            "def",
                                            "return",
                                            "#",
                                            '"',
                                            "'",
                                            "assert",
                                            "test_",
                                            "Test",
                                        )
                                    )
                                    or ">>>" in line
                                )
                            ):
                                break
                            result.append(line)
                        generated_code = "\n".join(result)

                # Fix missing syntax elements
                if generated_code and not generated_code.strip().endswith(":"):
                    if ":" not in generated_code:
                        generated_code += ":"
                if generated_code and "\n" not in generated_code:
                    generated_code += "\n    pass"

                # Fix indentation as last resort
                if generated_code:
                    lines = generated_code.split("\n")
                    fixed_lines = []
                    base_indent = None
                    for line in lines:
                        if line.strip():
                            if base_indent is None and line.startswith("def"):
                                base_indent = len(line) - len(line.lstrip())
                            if base_indent is not None:
                                stripped = (
                                    line[base_indent:]
                                    if line.startswith(" " * base_indent)
                                    else line
                                )
                                fixed_lines.append(
                                    "    " + stripped
                                    if stripped.strip()
                                    and not stripped.startswith("def")
                                    else stripped
                                )

                    fixed_code = "\n".join(fixed_lines)
                    logging.info(f"\nFinal fixed code:\n{fixed_code}\n")

                    # Try tests on fully fixed code - this is where we track errors
                    test_env = create_test_env()
                    if try_run_tests(
                        fixed_code, entry_point, test_code, test_env, error_tracker, idx
                    ):
                        correct_samples += 1
                        logging.info("✓ Sample passed all tests after full fixing")
                        continue

                logging.info("✗ Sample failed all test attempts")

            except Exception as e:
                error_tracker.add_error(idx, type(e).__name__)
                logging.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
                continue

        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)
        results.append(
            {
                "pass_at_k": pass_at_k,
                "error_stats": error_tracker.get_problem_stats(idx),
            }
        )
        logging.info(f"Problem {idx} pass@{k}: {pass_at_k:.2f}")
        logging.info(
            f"Error statistics for final attempts: {error_tracker.get_problem_stats(idx)}"
        )

    mean_pass_at_k = (
        sum(r["pass_at_k"] for r in results) / len(results) if results else 0
    )
    return mean_pass_at_k, results, error_tracker.get_total_stats()


def create_test_env():
    return {
        "__builtins__": __builtins__,
        "List": List,
        "Tuple": Tuple,
        "Any": Any,
        "Optional": Optional,
        "Union": Union,
        "Dict": Dict,
        "mean": statistics.mean,
        "candidate": None,
    }


def try_run_tests(
    code, entry_point, test_code, test_env, error_tracker=None, problem_idx=None
):
    """
    Try to run tests with error tracking for the final attempt.
    Returns (bool, str): (passed_tests, error_type)
    """
    try:
        # Add timeout for the entire code execution
        @timeout_decorator.timeout(10)  # 10 second timeout for entire test suite
        def run_code_with_timeout():
            try:
                exec(code, test_env)
            except IndentationError as e:
                return False, "IndentationError"
            except SyntaxError as e:
                return False, "SyntaxError"
            try:
                test_env["candidate"] = test_env[entry_point]
            except KeyError:
                return False, "RuntimeError"

            for test_case in test_code.split("\n"):
                test_case = test_case.strip()
                if test_case.startswith("assert"):
                    try:
                        exec(test_case, test_env)
                    except AssertionError:
                        return False, "AssertionError"
                    except TypeError:
                        return False, "TypeError"
                    except Exception as e:
                        return False, type(e).__name__
            return True, None

        result, error_type = run_code_with_timeout()
        if not result and error_tracker and problem_idx is not None:
            error_tracker.add_error(problem_idx, error_type)
        return result

    except timeout_decorator.TimeoutError:
        if error_tracker and problem_idx is not None:
            error_tracker.add_error(problem_idx, "TimeoutError")
        return False
    except Exception as e:
        if error_tracker and problem_idx is not None:
            error_tracker.add_error(problem_idx, type(e).__name__)
        return False


def assert_wrapper(condition, *args, **kwargs):
    """Custom assert function that just raises AssertionError on failure."""
    if not condition:
        raise AssertionError


def calculate_pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """
    Calculate pass@k metric from number of samples and correct solutions.

    Args:
        n_samples: Total number of samples generated
        n_correct: Number of correct samples
        k: k in pass@k metric

    Returns:
        float: pass@k probability
    """
    if n_correct < 0 or k < 0 or n_samples < 0:
        raise ValueError("Negative values not allowed")
    if n_correct > n_samples:
        raise ValueError("Cannot have more correct samples than total samples")
    if k > n_samples:
        raise ValueError("k cannot be greater than number of samples")

    if k == 0:
        return 1.0 if n_correct == n_samples else 0.0

    return 1.0 - math.comb(n_samples - n_correct, k) / math.comb(n_samples, k)


def main():
    # Model parameters
    model_name = "meta-llama/Llama-3.2-3B"  # need to add HF_TOKEN

    # Load dataset
    dataset = get_dataset("openai_humaneval", seed=42)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # No need to manually move model to device since we're using device_map="auto"
    # The model will be automatically placed on available GPUs

    # Evaluate
    logging.info("Starting evaluation...")

    accuracy, error_stats, detailed_results = evaluate_model(
        model, tokenizer, dataset, num_problems=10, n_samples=10, k=5
    )

    logging.info(f"\nFinal Accuracy: {accuracy:.2f}%")
    logging.info(f"Error Statistics:\n{json.dumps(error_stats, indent=2)}")

    # Save results
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(dataset),
        "error_statistics": error_stats,
        "detailed_results": detailed_results,
    }

    results_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
