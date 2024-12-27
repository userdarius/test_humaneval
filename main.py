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

logging.basicConfig(level=logging.INFO)


def extract_function(code_string, function_name):
    """Extract a function definition from a code string."""
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)

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
def execute_test_case(func_obj, test_case):
    """Execute a single test case and return True if it passes."""
    try:
        # Create a string buffer to capture stdout
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(test_case)
        return True
    except AssertionError:
        return False
    except Exception as e:
        logging.error(f"Error executing test case: {e}")
        return False


def evaluate_model(model, tokenizer, dataset, num_samples=10):
    correct = 0
    total = 0

    for item in tqdm(dataset[:num_samples]):
        question = item["question"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]

        # Prepare prompt
        prompt = f"""Write Python code to solve this problem. Only provide the code without any explanations.

{question}

Answer:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = response[len(prompt) :].strip()

        try:
            # Extract the function from the generated code
            func_code = extract_function(generated_code, entry_point)
            if func_code is None:
                logging.info(f"Could not find function {entry_point} in generated code")
                total += 1
                continue

            # Create a new namespace for the function
            namespace = {}

            # Execute the function definition
            exec(func_code, namespace)

            # Get the function object
            func_obj = namespace[entry_point]

            # Prepare the test environment
            test_env = {entry_point: func_obj, "assert": assert_wrapper}

            # Execute test cases
            test_cases = test_code.split("\n")
            test_results = []

            for test_case in test_cases:
                if test_case.strip().startswith("assert"):
                    try:
                        result = execute_test_case(func_obj, test_case)
                        test_results.append(result)
                    except timeout_decorator.TimeoutError:
                        test_results.append(False)
                        logging.warning("Test case timed out")

            # Consider the solution correct if all test cases pass
            if test_results and all(test_results):
                correct += 1

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
    model_name = "meta-llama/Llama-2-3b"  # You'll need HF access token

    # Load dataset
    dataset = get_dataset("openai_humaneval", seed=42)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate
    logging.info("Starting evaluation...")
    accuracy = evaluate_model(model, tokenizer, dataset)

    logging.info(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
