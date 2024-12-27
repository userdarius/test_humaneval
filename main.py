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
        if code_string.startswith(f'def {function_name}'):
            return code_string
        
        # Try parsing as AST
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            # Try adding a newline at the start in case indentation is wrong
            try:
                tree = ast.parse('\n' + code_string)
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
        # Create a string buffer to capture stdout
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            # Execute in the provided test environment
            exec(test_case, test_env)
        return True
    except AssertionError:
        return False
    except Exception as e:
        logging.error(f"Error executing test case: {e}")
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

        # Improved prompt with stricter instructions and function signature
        prompt = f"""Complete the following Python function. Only write the function implementation, nothing else.

{question}

Complete the implementation below:
def {entry_point}"""

        try:
            encoded_input = tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=False,  # Disable sampling for more deterministic output
                    temperature=0.1,
                    num_beams=5,      # Use beam search
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the function implementation
            if "def " + entry_point in response:
                generated_code = response[response.find("def " + entry_point):]
                # Remove anything after the function (like test cases)
                if "\n\n" in generated_code:
                    generated_code = generated_code.split("\n\n")[0]
            else:
                generated_code = response[len(prompt):].strip()

            logging.info(f"\nGenerated code:\n{generated_code}\n")
            
            if not generated_code or "pass" in generated_code:
                logging.error("Invalid or empty implementation")
                total += 1
                continue

            # Create test environment
            test_env = {
                '__builtins__': __builtins__,
                'List': List,
                'Tuple': Tuple,
                'Any': Any,
                'Optional': Optional,
                'Union': Union,
                'Dict': Dict,
                'candidate': None  # Add candidate for compatibility with test cases
            }

            # Execute function definition
            exec(generated_code, test_env)
            
            # Set the candidate for test cases
            test_env['candidate'] = test_env[entry_point]

            # Execute test cases
            test_results = []
            for test_case in test_code.split('\n'):
                if test_case.strip().startswith('assert'):
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
        "num_samples": len(dataset)
    }
    
    results_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
