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


def evaluate_model(model, tokenizer, dataset, num_samples=5):
    correct = 0
    total = 0

    # Get model's device
    device = next(model.parameters()).device

    for item in tqdm(dataset[:num_samples]):
        question = item["question"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]

        # Improved prompt with explicit structure and example
        prompt = f"""Write a complete Python function implementation. Follow this exact structure:

1. Function signature (with type hints, no docstring)
2. Function body with implementation
3. Return statement

Here is the function to implement:
{question}

Begin your implementation with:
def {entry_point}"""

        try:
            encoded_input = tokenizer(prompt, return_tensors="pt", truncation=True)
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
                    temperature=0.1,  # Even lower temperature for more focused output
                    top_p=0.9,
                    num_beams=5,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Add repetition penalty
                    length_penalty=1.0,      # Add length penalty
                    min_length=50,           # Ensure minimum length
                    no_repeat_ngram_size=2,  # Prevent repetition of n-grams
                    early_stopping=False,     # Don't stop early
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the function implementation
            if "def " + entry_point in response:
                generated_code = response[response.find("def " + entry_point):]
                # Remove anything after a double newline or common endings
                for ending in ["\n\n", "\n# Test", "\n# Example", "\nif __name__"]:
                    if ending in generated_code:
                        generated_code = generated_code.split(ending)[0]
            else:
                generated_code = response[len(prompt):].strip()

            # Ensure the function has a body by checking for colon and indentation
            if not generated_code.strip().endswith(":"):
                # Try to add the missing colon
                if not ":" in generated_code:
                    generated_code += ":"

            # Add a basic implementation if no body is present
            if not "\n" in generated_code:
                generated_code += "\n    pass"

            logging.info(f"\nGenerated code:\n{generated_code}\n")

            # After generating code
            if not generated_code.strip().endswith(":"):
                logging.error("Incomplete function definition")
                continue

            if "return" not in generated_code:
                logging.error("Missing return statement")
                continue

            # Fix indentation in the generated code
            lines = generated_code.split("\n")
            fixed_lines = []
            base_indent = None
            for line in lines:
                if line.strip():
                    if base_indent is None and line.startswith("def"):
                        base_indent = len(line) - len(line.lstrip())
                    if base_indent is not None:
                        # Remove base indentation and add 4 spaces for proper indentation
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

            # Create test environment
            test_env = {
                "__builtins__": __builtins__,
                "List": List,
                "Tuple": Tuple,
                "Any": Any,
                "Optional": Optional,
                "Union": Union,
                "Dict": Dict,
                "candidate": None,  # Add candidate for compatibility with test cases
            }

            # Execute function definition
            try:
                exec(fixed_code, test_env)
            except Exception as e:
                logging.error(f"Error in function definition: {e}")
                total += 1
                continue

            # Set the candidate for test cases
            test_env["candidate"] = test_env[entry_point]

            # Execute test cases with fixed indentation
            test_results = []
            for test_case in test_code.split("\n"):
                test_case = test_case.strip()  # Remove any indentation
                if test_case.startswith("assert"):
                    try:
                        exec(test_case, test_env)
                        test_results.append(True)
                    except Exception as e:
                        logging.error(f"Test failed: {e}")
                        test_results.append(False)

            if test_results and all(test_results):
                correct += 1
                logging.info("✓ All tests passed")
            else:
                logging.info("✗ Some tests failed")

            total += 1

        except Exception as e:
            logging.error(f"Error: {e}")
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy


def assert_wrapper(condition, *args, **kwargs):
    """Custom assert function that just raises AssertionError on failure."""
    if not condition:
        raise AssertionError


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
    accuracy = evaluate_model(model, tokenizer, dataset)

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
