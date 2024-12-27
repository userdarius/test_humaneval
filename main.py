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


def evaluate_model(model, tokenizer, dataset, num_problems=5, n_samples=10, k=1):
    """
    Evaluate model using pass@k metric.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset: The evaluation dataset
        num_problems: Number of problems to evaluate
        n_samples: Number of samples to generate per problem
        k: k in pass@k metric
    """
    results = []
    device = next(model.parameters()).device

    for item in tqdm(dataset[:num_problems]):
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
                        temperature=0.2,  # Higher temperature for more diversity
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

                logging.info(f"\nGenerated code:\n{response}\n")

                if "def " + entry_point in response:
                    generated_code = response[response.find("def " + entry_point) :]
                    for ending in ["\n\n", "\n# Test", "\n# Example", "\nif __name__"]:
                        if ending in generated_code:
                            generated_code = generated_code.split(ending)[0]
                else:
                    generated_code = response[len(question) :].strip()

                if not generated_code.strip().endswith(":"):
                    if not ":" in generated_code:
                        generated_code += ":"

                if not "\n" in generated_code:
                    generated_code += "\n    pass"

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

                logging.info(f"\nFixed code:\n{fixed_code}\n")

                test_env = {
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

                exec(fixed_code, test_env)
                test_env["candidate"] = test_env[entry_point]

                test_results = []
                for test_case in test_code.split("\n"):
                    test_case = test_case.strip()
                    if test_case.startswith("assert"):
                        try:
                            exec(test_case, test_env)
                            test_results.append(True)
                        except Exception:
                            test_results.append(False)

                if test_results and all(test_results):
                    correct_samples += 1
                    logging.info("✓ Sample passed all tests")
                else:
                    logging.info("✗ Sample failed some tests")

            except Exception as e:
                logging.error(f"Error: {e}")
                continue

        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)
        results.append(pass_at_k)
        logging.info(f"Problem pass@{k}: {pass_at_k:.2f}")

    mean_pass_at_k = sum(results) / len(results) if results else 0
    return mean_pass_at_k


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
    model_name = "meta-llama/Llama-3.2-3b"  # need to add HF_TOKEN

    # Load dataset
    dataset = get_dataset("openai_humaneval", seed=42)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # No need to manually move model to device since we're using device_map="auto"
    # The model will be automatically placed on available GPUs

    # Evaluate
    logging.info("Starting evaluation...")
    accuracy = evaluate_model(model, tokenizer, dataset, num_problems=5, n_samples=10, k=1)

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
