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
    device = next(model.parameters()).device

    for item in tqdm(dataset[:num_samples]):
        question = item["question"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]

        prompt = f"""Write a complete Python function implementation including the body and return statement. The function MUST include a complete implementation after the signature.

Function to implement:
{question}

Your complete implementation should follow this structure:
def {entry_point}
    # Implementation here
    # Must include actual code, not just signature
    return result

Begin your implementation:"""

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
                    temperature=0.2,  # Lower temperature for more focused output
                    top_p=0.95,
                    num_beams=5,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    min_length=100,  # Increase minimum length
                    max_length=2048,  # Increase maximum length
                    no_repeat_ngram_size=3,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the function implementation
            if "def " + entry_point in response:
                generated_code = response[response.find("def " + entry_point) :]
                # Clean up the generated code
                for ending in ["\n\n", "\n# Test", "\n# Example", "\nif __name__"]:
                    if ending in generated_code:
                        generated_code = generated_code.split(ending)[0].strip()
            else:
                continue

            # Verify the function has a complete implementation
            if not generated_code.strip().endswith(":"):
                generated_code += ":"

            # Ensure there's a function body
            if not "\n" in generated_code or "return" not in generated_code.lower():
                logging.error(
                    "Incomplete implementation - missing body or return statement"
                )
                continue

            # Fix indentation
            lines = generated_code.split("\n")
            fixed_lines = []
            first_line = True
            for line in lines:
                if first_line:
                    fixed_lines.append(line)
                    first_line = False
                else:
                    # Ensure proper indentation for function body
                    if line.strip():
                        fixed_lines.append("    " + line.lstrip())

            fixed_code = "\n".join(fixed_lines)
            logging.info(f"\nProcessed code:\n{fixed_code}\n")

            # Create test environment
            test_env = {
                "__builtins__": __builtins__,
                "List": List,
                "Optional": Optional,
                "Union": Union,
                "Dict": Dict,
                "Tuple": Tuple,
                "Any": Any,
                "candidate": None,
            }

            # Execute function definition
            try:
                exec(fixed_code, test_env)
            except Exception as e:
                logging.error(f"Error in function definition: {e}")
                continue

            # Set up for testing
            test_env["candidate"] = test_env[entry_point]

            # Run tests
            all_tests_passed = True
            for test_case in test_code.split("\n"):
                test_case = test_case.strip()
                if test_case.startswith("assert"):
                    try:
                        exec(test_case, test_env)
                    except Exception as e:
                        logging.error(f"Test failed: {e}")
                        all_tests_passed = False
                        break

            if all_tests_passed:
                correct += 1
                logging.info("✓ All tests passed")
            else:
                logging.info("✗ Some tests failed")

            total += 1

        except Exception as e:
            logging.error(f"Error: {e}")
            total += 1
            continue

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
