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

logging.basicConfig(level=logging.INFO)


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
    except Exception as e:
        logging.error(f"Error executing test case: {type(e).__name__}: {str(e)}")
        return False


def evaluate_model(model, tokenizer, dataset, num_problems, n_samples, k):
    """
    Evaluate the model on the dataset.
    """
    results = []
    device = next(model.parameters()).device
    sampled_problems = np.random.choice(len(dataset), size=num_problems, replace=False)

    for idx in tqdm(sampled_problems):
        item = dataset[idx]
        question = item["question"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]
        correct_samples = 0

        for _ in range(n_samples):
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

                # Extract function if raw response failed
                if "def " + entry_point in response:
                    start = response.find("def " + entry_point)
                    generated_code = response[start:]

                    # First try AST parsing
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

                            # Track docstring state
                            if '"""' in line or "'''" in line:
                                docstring_delim += line.count('"""') + line.count("'''")
                                in_docstring = docstring_delim % 2 != 0

                            # Check for end of function
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
                else:
                    generated_code = ""

                logging.info(f"\nExtracted code:\n{generated_code}\n")

                # Try tests on extracted function
                test_env = create_test_env()
                if try_run_tests(generated_code, entry_point, test_code, test_env):
                    correct_samples += 1
                    logging.info("✓ Sample passed all tests after function extraction")
                    continue

                # Add missing syntax elements if needed
                if not generated_code.strip().endswith(":"):
                    if not ":" in generated_code:
                        generated_code += ":"
                if not "\n" in generated_code:
                    generated_code += "\n    pass"

                logging.info(f"\nMissing syntax code fixed:\n{generated_code}\n")

                # Try tests after syntax fixing
                test_env = create_test_env()
                if try_run_tests(generated_code, entry_point, test_code, test_env):
                    correct_samples += 1
                    logging.info("✓ Sample passed all tests after syntax fixing")
                    continue

                # Fix indentation as last resort
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
                                if stripped.strip() and not stripped.startswith("def")
                                else stripped
                            )

                fixed_code = "\n".join(fixed_lines)
                logging.info(f"\nFinal fixed code:\n{fixed_code}\n")

                # Try tests on fully fixed code
                test_env = create_test_env()
                if try_run_tests(fixed_code, entry_point, test_code, test_env):
                    correct_samples += 1
                    logging.info("✓ Sample passed all tests after full fixing")
                    continue

                logging.info("✗ Sample failed all test attempts")

            except Exception as e:
                logging.error(f"Error: {e}")
                continue

        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)
        results.append(pass_at_k)
        logging.info(f"Problem pass@{k}: {pass_at_k:.2f}")

    mean_pass_at_k = sum(results) / len(results) if results else 0
    return mean_pass_at_k


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


def try_run_tests(code, entry_point, test_code, test_env):
    try:
        exec(code, test_env)
        test_env["candidate"] = test_env[entry_point]

        for test_case in test_code.split("\n"):
            test_case = test_case.strip()
            if test_case.startswith("assert"):
                try:
                    exec(test_case, test_env)
                except Exception as e:
                    logging.error(f"Test failed: {test_case}")
                    logging.error(f"Error: {str(e)}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
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
    accuracy = evaluate_model(
        model, tokenizer, dataset, num_problems=100, n_samples=10, k=5
    )

    logging.info(f"\nFinal Accuracy: {accuracy:.2f}%")

    # Save results
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(dataset),
    }

    results_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
