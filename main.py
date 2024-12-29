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
from model import EntailmentDeberta
from scores import (
    get_semantic_ids,
    cluster_assignment_entropy,
    predictive_entropy,
    predictive_entropy_rao,
    context_entails_response,
)

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


def extract_function_body(code_string):
    """Extract just the function body, excluding signature and docstring."""
    try:
        # First, normalize any initial indentation
        lines = code_string.split("\n")
        min_indent = float("inf")

        # Find minimum indentation of non-empty lines
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # Normalize the indentation
        if min_indent != float("inf"):
            lines = [line[min_indent:] if line.strip() else line for line in lines]
        code_string = "\n".join(lines)

        # Parse the code
        tree = ast.parse(code_string)

        # Find the first function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip the function definition line and docstring if present
                body = node.body
                if (
                    len(body) > 0
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Str)
                ):
                    body = body[1:]  # Skip docstring

                # Get just the actual implementation lines
                result = []
                for b in body:
                    if isinstance(b, ast.Return):
                        result.append(ast.unparse(b).strip())
                    else:
                        result.append(ast.unparse(b))

                return "\n".join(result)

        return ""  # Return empty string if no function found
    except Exception as e:
        logging.error(f"Error extracting function body: {e}")
        return ""  # Return empty string on error


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


def evaluate_model(
    model, tokenizer, dataset, num_problems, n_samples, k, entailment_model
):
    """
    Evaluate the model on the dataset with error tracking and semantic uncertainty metrics.
    Computes metrics for all generated solutions, not just passing ones.
    """
    results = []
    error_tracker = ErrorTracker()

    device = next(model.parameters()).device
    logging.info(f"\nStarting evaluation on device: {device}")

    for idx in tqdm(range(num_problems)):
        logging.info(f"\n{'='*50}")
        logging.info(f"Problem {idx}")

        item = dataset[idx]
        question = item["question"]
        canonical_solution = item["canonical_solution"]
        entry_point = item["entry_point"]
        test_code = item["test_code"]

        logging.info(f"Question length: {len(question)}")
        logging.info(f"Canonical solution length: {len(canonical_solution)}")

        # Track all solutions and their outcomes
        all_solutions = []  # Store all solutions regardless of pass/fail
        all_log_probs = []  # Store all log probs
        solution_outcomes = []  # Track pass/fail for each solution
        correct_samples = 0

        for sample_idx in range(n_samples):
            error_tracker.increment_total(idx)

            try:
                # Generate solution
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
                        num_beams=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        min_length=50,
                        no_repeat_ngram_size=2,
                        early_stopping=False,
                    )

                # Calculate log probability with validation
                log_prob = 0.0
                if hasattr(outputs, "scores") and outputs.scores:
                    scores = outputs.scores
                    generated_ids = outputs.sequences[0]
                    for step, score in enumerate(scores):
                        if isinstance(score, tuple):
                            score = score[0]
                        step_log_probs = torch.log_softmax(score, dim=-1)
                        if step < len(generated_ids) - 1:
                            token = generated_ids[step + 1]
                            log_prob_step = step_log_probs[0, token].item()
                            if not np.isfinite(log_prob_step):
                                log_prob_step = -1e3
                            log_prob += log_prob_step

                # Get generated code
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                logging.info(f"\nGenerated solution {sample_idx + 1}:")
                logging.debug(f"{response[:200]}...")

                # Extract and fix code
                final_code = response
                if "def " + entry_point in response:
                    final_code = extract_and_fix_code(response, entry_point)

                # Test the solution
                test_env = create_test_env()
                passed_tests = try_run_tests(
                    final_code, entry_point, test_code, test_env, error_tracker, idx
                )

                if passed_tests:
                    correct_samples += 1
                    logging.info("✓ Solution passed tests")
                else:
                    logging.info("✗ Solution failed tests")

                # Store solution and outcome
                all_solutions.append(final_code)
                all_log_probs.append(log_prob)
                solution_outcomes.append(passed_tests)

            except Exception as e:
                error_tracker.add_error(idx, type(e).__name__)
                logging.error(
                    f"Error generating solution: {type(e).__name__}: {str(e)}"
                )
                continue

        # Calculate semantic metrics for all solutions
        semantic_metrics = {}
        if all_solutions:
            logging.info(
                f"\nCalculating semantic metrics for all {len(all_solutions)} solutions"
            )

            # Get semantic clusters
            semantic_ids = get_semantic_ids(all_solutions, entailment_model)
            num_clusters = len(set(semantic_ids))
            logging.info(f"Number of semantic clusters: {num_clusters}")

            # Calculate entropies
            semantic_entropy = cluster_assignment_entropy(semantic_ids)
            pred_entropy = predictive_entropy(all_log_probs)
            pred_entropy_rao = predictive_entropy_rao(all_log_probs)

            logging.info(f"Semantic entropy: {semantic_entropy:.3f}")
            logging.info(f"Predictive entropy: {pred_entropy:.3f}")

            # Calculate alignments
            canonical_body = extract_function_body(canonical_solution)
            generated_bodies = [
                extract_function_body(sol)
                for sol in all_solutions
                if extract_function_body(sol)
            ]

            if generated_bodies:
                canonical_alignment = context_entails_response(
                    canonical_body, generated_bodies, entailment_model
                )
                reverse_alignment = context_entails_response(
                    canonical_body, generated_bodies, entailment_model
                )
                bidirectional = (canonical_alignment + reverse_alignment) / 2

                logging.info(f"Canonical alignment: {canonical_alignment:.3f}")
                logging.info(f"Bidirectional alignment: {bidirectional:.3f}")

                # Additional metrics for passing vs failing solutions
                passing_solutions = [
                    sol
                    for sol, passed in zip(all_solutions, solution_outcomes)
                    if passed
                ]
                failing_solutions = [
                    sol
                    for sol, passed in zip(all_solutions, solution_outcomes)
                    if not passed
                ]

                semantic_metrics = {
                    "semantic_entropy": semantic_entropy,
                    "predictive_entropy": pred_entropy,
                    "predictive_entropy_rao": pred_entropy_rao,
                    "num_semantic_clusters": num_clusters,
                    "num_solutions": len(all_solutions),
                    "num_passing": len(passing_solutions),
                    "num_failing": len(failing_solutions),
                    "canonical_alignment": canonical_alignment,
                    "reverse_alignment": reverse_alignment,
                    "bidirectional_alignment": bidirectional,
                }

        # Calculate pass@k
        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)

        # Store results
        results.append(
            {
                "problem_id": idx,
                "pass_at_k": pass_at_k,
                "error_stats": error_tracker.get_problem_stats(idx),
                "semantic_metrics": semantic_metrics,
                "solution_outcomes": solution_outcomes,
            }
        )

        logging.info(f"\nProblem {idx} Summary:")
        logging.info(f"pass@{k}: {pass_at_k:.2f}")
        logging.info(f"Correct samples: {correct_samples}/{n_samples}")
        if semantic_metrics:
            logging.info(
                f"Semantic clusters: {semantic_metrics['num_semantic_clusters']}"
            )
            logging.info(
                f"Passing/Failing: {semantic_metrics['num_passing']}/{semantic_metrics['num_failing']}"
            )

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)

    return aggregate_metrics, results, error_tracker.get_total_stats()


def extract_and_fix_code(response, entry_point):
    """Helper function to extract and fix generated code."""
    start = response.find("def " + entry_point)
    code = response[start:]

    try:
        # Try AST parsing
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                code = "\n".join(code.split("\n")[: node.end_lineno])
                break
    except SyntaxError:
        # Fallback to manual parsing
        code = manual_code_extraction(code)

    return fix_code_formatting(code)


def manual_code_extraction(code):
    """Extract code using manual parsing."""
    lines = code.split("\n")
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
                    ("def", "return", "#", '"', "'", "assert", "test_", "Test")
                )
                or ">>>" in line
            )
        ):
            break
        result.append(line)

    return "\n".join(result)


def fix_code_formatting(code):
    """Fix code formatting issues."""
    if not code.strip().endswith(":"):
        if ":" not in code:
            code += ":"
    if "\n" not in code:
        code += "\n    pass"

    # Fix indentation
    lines = code.split("\n")
    fixed_lines = []
    base_indent = None

    for line in lines:
        if line.strip():
            if base_indent is None and line.startswith("def"):
                base_indent = len(line) - len(line.lstrip())
            if base_indent is not None:
                stripped = (
                    line[base_indent:] if line.startswith(" " * base_indent) else line
                )
                fixed_lines.append(
                    "    " + stripped
                    if stripped.strip() and not stripped.startswith("def")
                    else stripped
                )

    return "\n".join(fixed_lines)


def calculate_aggregate_metrics(results):
    """Calculate aggregate metrics across all problems."""
    return {
        "mean_pass_at_k": np.mean([r["pass_at_k"] for r in results]),
        "mean_semantic_entropy": np.mean(
            [
                r["semantic_metrics"].get("semantic_entropy", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
        "mean_predictive_entropy": np.mean(
            [
                r["semantic_metrics"].get("predictive_entropy", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
        "mean_canonical_alignment": np.mean(
            [
                r["semantic_metrics"].get("canonical_alignment", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
        "mean_bidirectional_alignment": np.mean(
            [
                r["semantic_metrics"].get("bidirectional_alignment", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
        "total_solutions": sum(
            [
                r["semantic_metrics"].get("num_solutions", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
        "total_passing": sum(
            [
                r["semantic_metrics"].get("num_passing", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
        "total_failing": sum(
            [
                r["semantic_metrics"].get("num_failing", 0)
                for r in results
                if r["semantic_metrics"]
            ]
        ),
    }


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
    logging.info("Loading dataset...")
    dataset = get_dataset("openai_humaneval", seed=42)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load entailment model
    logging.info("Loading entailment model...")
    entailment_model = EntailmentDeberta()

    # No need to manually move model to device since we're using device_map="auto"
    # The model will be automatically placed on available GPUs

    # Evaluate
    logging.info("Starting evaluation...")

    aggregate_metrics, detailed_results, error_stats = evaluate_model(
        model,
        tokenizer,
        dataset,
        num_problems=10,
        n_samples=10,
        k=5,
        entailment_model=entailment_model,
    )

    # Print aggregate metrics
    logging.info("\nFinal Results:")
    logging.info(f"Mean pass@k: {aggregate_metrics['mean_pass_at_k']:.2f}")
    logging.info(
        f"Mean semantic entropy: {aggregate_metrics['mean_semantic_entropy']:.2f}"
    )
    logging.info(
        f"Mean predictive entropy: {aggregate_metrics['mean_predictive_entropy']:.2f}"
    )
    logging.info(
        f"Mean canonical alignment: {aggregate_metrics['mean_canonical_alignment']:.2f}"
    )
    logging.info(f"Error Statistics:\n{json.dumps(error_stats, indent=2)}")

    # Save results
    results = {
        "model_name": model_name,
        "aggregate_metrics": aggregate_metrics,
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
