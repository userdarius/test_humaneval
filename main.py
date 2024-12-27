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


def evaluate_model(model, tokenizer, dataset, num_samples=10):
    correct = 0
    total = 0
    
    # Get model's device
    device = next(model.parameters()).device

    for item in tqdm(dataset[:num_samples]):
        question = item["question"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]

        # Improved prompt with clear instructions and example format
        prompt = f"""Write a complete Python function to solve this programming problem. Include the full implementation.

Problem:
{question}

Write your solution below, starting with the function definition 'def {entry_point}'. Do not include any comments or test cases.

Example format:
def example_function(arg1: type1, arg2: type2) -> return_type:
    # Implementation
    return result

Your solution:
def {entry_point}"""

        # Generate response
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
                    do_sample=True,
                    temperature=0.2,  # Lower temperature for more focused outputs
                    top_p=0.95,
                    top_k=50,        # Add top_k sampling
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    min_new_tokens=10,  # Ensure some minimum output
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract everything after "Your solution:"
            if "Your solution:" in response:
                generated_code = response.split("Your solution:")[-1].strip()
            else:
                generated_code = response[len(prompt):].strip()
            
            # Clean up the generated code
            generated_code = generated_code.strip('`').strip()
            
            # Log the full interaction for debugging
            logging.info(f"\nPrompt:\n{prompt}\n")
            logging.info(f"Raw response:\n{response}\n")
            logging.info(f"Extracted code:\n{generated_code}\n")
            
            if not generated_code or not generated_code.startswith("def"):
                logging.error("Invalid code generation")
                total += 1
                continue

            # Extract the function from the generated code
            func_code = extract_function(generated_code, entry_point)
            if func_code is None:
                # Try to use the entire generated code if it starts with 'def'
                if generated_code.strip().startswith('def'):
                    func_code = generated_code
                else:
                    logging.info(f"Could not find function {entry_point} in generated code")
                    total += 1
                    continue

            # Create namespace with all necessary imports and functions
            namespace = {
                'List': list,
                'Optional': Optional,
                'Union': Union,
                'Dict': Dict,
                'Tuple': Tuple,
                'Any': Any,
                '__builtins__': __builtins__,
                'mean': lambda x: sum(x) / len(x),  # Add mean function
                'math': math,
                'statistics': statistics,
                'numpy': np,
            }
            
            # Add all necessary imports
            exec('''
from typing import List, Optional, Union, Dict, Tuple, Any
import math
import statistics
import numpy as np
''', namespace)
            
            # Execute the function definition
            exec(func_code, namespace)

            # Get the function object
            func_obj = namespace[entry_point]

            # Prepare test environment with same imports
            test_env = namespace.copy()
            test_env.update({
                entry_point: func_obj,
                "assert": assert_wrapper,
            })

            # Execute test cases with proper indentation
            test_cases = []
            for line in test_code.split("\n"):
                if line.strip().startswith("assert"):
                    # Remove any indentation from assert statements
                    test_cases.append(line.strip())
                else:
                    test_cases.append(line)
            
            test_code = "\n".join(test_cases)
            test_results = []

            for test_case in test_cases:
                if test_case.strip().startswith("assert"):
                    try:
                        result = execute_test_case(func_obj, test_case.strip(), test_env)
                        test_results.append(result)
                    except timeout_decorator.TimeoutError:
                        test_results.append(False)
                        logging.warning("Test case timed out")

            # Consider the solution correct if all test cases pass
            if test_results and all(test_results):
                correct += 1
                logging.info("✓ All test cases passed")
            else:
                logging.info("✗ Some test cases failed")
                # Log which test cases failed
                for i, (test_case, result) in enumerate(zip(test_cases, test_results)):
                    if test_case.strip().startswith("assert"):
                        status = "✓" if result else "✗"
                        logging.info(f"{status} Test {i+1}: {test_case.strip()}")

            total += 1

            logging.info(f"\nQuestion: {question[:100]}...")
            logging.info(f"Generated Code:\n{func_code}")
            logging.info(f"Test Results: {test_results}")

        except Exception as e:
            logging.error(f"Error processing response: {e}")
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
