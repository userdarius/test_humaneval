import torch
from data import get_dataset
from model import speculative_sampling, load_approx_and_target_model_and_tokenizer
from tqdm import tqdm
import ast
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
import logging
import gc
import os
import sys


logging.basicConfig(level=logging.INFO)

# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_experiment_dir():
    """Create a timestamped directory for the current experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"humaneval_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def setup_logging(experiment_dir):
    """Configure logging with detailed formatting and both file and console handlers"""
    log_filename = os.path.join(experiment_dir, "humaneval.log")

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename


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


def extract_function_body(code_string: str) -> Optional[str]:
    """
    Extract just the function body focusing on the actual implementation.
    Handles both docstrings and implementation code more robustly.
    """
    try:
        # First try AST parsing for clean code
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get the function body lines
                    lines = code_string.split("\n")
                    # Skip function definition line
                    body_lines = lines[node.body[0].lineno - 1 : node.end_lineno]
                    # Remove docstring if present
                    if isinstance(node.body[0], ast.Expr) and isinstance(
                        node.body[0].value, ast.Str
                    ):
                        body_lines = body_lines[
                            node.body[1].lineno - node.body[0].lineno :
                        ]
                    return "\n".join(body_lines).strip()
        except (SyntaxError, AttributeError):
            pass

        # Fallback: Manual parsing
        lines = code_string.split("\n")
        content_lines = []
        in_docstring = False
        implementation_started = False
        docstring_delim = 0

        for line in lines:
            stripped = line.strip()

            # Handle docstring boundaries
            if '"""' in line or "'''" in line:
                docstring_delim += line.count('"""') + line.count("'''")
                in_docstring = docstring_delim % 2 != 0
                continue

            # Skip if we're in a docstring
            if in_docstring:
                continue

            # Skip function definition and empty lines
            if stripped.startswith("def ") or not stripped:
                continue

            # Skip comment lines and doctest examples
            if stripped.startswith("#") or ">>>" in line:
                continue

            # Skip common non-implementation markers
            if stripped.startswith(("@", "class ", "if __name__")):
                continue

            # This is likely implementation code
            if not implementation_started:
                if stripped and line[0].isspace():  # Check for indentation
                    implementation_started = True

            if implementation_started:
                if not line[0].isspace():  # End of function
                    break
                content_lines.append(line)

        if not content_lines:
            return None

        # Join implementation lines
        implementation = "\n".join(content_lines)

        return implementation.strip()

    except Exception as e:
        logging.error(f"Error in function body extraction: {e}")
        return None


@timeout_decorator.timeout(10)  # 5 second timeout for execution
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


def calculate_sequence_log_prob(outputs, generated_ids, tokenizer):
    """
    Calculate normalized log probability for a generated sequence.

    Args:
        outputs: Model generation outputs containing scores
        generated_ids: Generated token IDs
        tokenizer: Tokenizer for handling special tokens

    Returns:
        float: Normalized log probability of the sequence
    """
    if not hasattr(outputs, "scores") or not outputs.scores:
        return 0.0

    scores = outputs.scores
    log_prob = 0.0
    sequence_length = 0

    # Get indices of non-padding tokens
    non_pad_indices = (generated_ids != tokenizer.pad_token_id).nonzero().squeeze(-1)
    if len(non_pad_indices) == 0:
        return 0.0

    # Only consider tokens after the prompt (input)
    start_idx = non_pad_indices[0].item()

    for step, score in enumerate(scores):
        if isinstance(score, tuple):
            score = score[0]

        # Get log probabilities for current step
        step_log_probs = torch.log_softmax(score, dim=-1)

        # Only include if we're past the prompt
        if step + start_idx + 1 < len(generated_ids):
            token = generated_ids[step + start_idx + 1]

            # Skip padding tokens
            if token == tokenizer.pad_token_id:
                continue

            log_prob_step = step_log_probs[0, token].item()

            # Handle potential numerical instabilities
            if not np.isfinite(log_prob_step):
                log_prob_step = -100.0  # Less extreme default value

            log_prob += log_prob_step
            sequence_length += 1

    # Normalize by sequence length to get per-token log probability
    if sequence_length > 0:
        normalized_log_prob = log_prob / sequence_length
    else:
        normalized_log_prob = 0.0

    # Clip to reasonable range
    return np.clip(normalized_log_prob, -100.0, 0.0)


def evaluate_model(
    approx_model,
    target_model,
    tokenizer,
    dataset,
    num_problems,
    n_samples,
    k,
    entailment_model,
    experiment_dir,
):
    """
    Enhanced evaluation function with additional metrics and visualization
    """
    results = []
    error_tracker = ErrorTracker()

    for idx in tqdm(range(num_problems)):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"\n{'='*50}")
        logging.info(f"Problem {idx}")

        problem = dataset[idx]
        metrics = evaluate_problem(
            approx_model,
            target_model,
            tokenizer,
            problem,
            idx,
            n_samples,
            k,
            entailment_model,
            error_tracker,
        )

        if metrics:
            results.append(metrics)

            # Log metrics
            logging.info(f"\nProblem {idx} Results:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{key}: {value:.4f}")

        # Save detailed results
        results_file = os.path.join(experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Calculate and return aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)
    return aggregate_metrics, results, error_tracker.get_total_stats()


def extract_and_fix_function(code, entry_point):
    """Helper function to extract and fix a function definition"""
    try:
        # Try AST parsing
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                end = node.end_lineno
                return "\n".join(code.split("\n")[:end])
    except SyntaxError:
        # Fallback: manual parsing
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

        code = "\n".join(result)

        # Fix missing syntax elements
        if code and not code.strip().endswith(":"):
            if ":" not in code:
                code += ":"
        if code and "\n" not in code:
            code += "\n    pass"

        # Fix indentation
        if code:
            lines = code.split("\n")
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
            return "\n".join(fixed_lines)

    return code


def evaluate_problem(
    approx_model,
    target_model,
    tokenizer,
    problem,
    idx,
    n_samples,
    k,
    entailment_model,
    error_tracker,
):
    """Evaluate a single problem with enhanced metrics"""
    question = problem["question"]
    canonical_solution = problem["canonical_solution"]
    entry_point = problem["entry_point"]
    test_code = problem["test_code"]

    # Store all solutions and their scores
    raw_solutions = []
    processed_solutions = []
    solution_log_probs = []
    correct_samples = 0

    try:
        inputs = tokenizer(
            question,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(target_model.device)
        logging.debug(f"Tokenized input length: {inputs['input_ids'].size()}")
    except Exception as e:
        logging.error(f"Tokenization failed: {str(e)}")
        raise

    try:
        # Generate multiple samples
        for _ in range(n_samples):
            # Generate one solution using speculative sampling
            output_ids, token_log_probs = speculative_sampling(
                prefix=inputs.input_ids,
                approx_model=approx_model,
                target_model=target_model,
                max_len=256,
                gamma=4,
                temperature=0.6,
                top_k=0,
                top_p=0,
            )

            # Decode the generated tokens into text
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Calculate average log probability
            avg_log_prob = np.mean(token_log_probs) if token_log_probs else 0.0

            error_tracker.increment_total(idx)
            raw_solutions.append(response)
            scaled_log_prob = np.clip(avg_log_prob, -10.0, 0.0)
            solution_log_probs.append(scaled_log_prob)

            # Extract and process function
            if "def " + entry_point in response:
                generated_code = extract_and_fix_function(
                    response[response.find("def " + entry_point) :], entry_point
                )
                if generated_code:
                    processed_solutions.append(generated_code)

                # Run tests and track correctness
                test_env = create_test_env()
                if try_run_tests(response, entry_point, test_code, test_env):
                    correct_samples += 1
                    continue

                if generated_code:
                    test_env = create_test_env()
                    if try_run_tests(
                        generated_code,
                        entry_point,
                        test_code,
                        test_env,
                        error_tracker,
                        idx,
                    ):
                        correct_samples += 1
                        continue

        # Calculate semantic metrics
        semantic_metrics = calculate_semantic_metrics(
            processed_solutions,
            canonical_solution,
            solution_log_probs,
            entailment_model,
        )

        # Calculate pass@k
        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)

        # Additional code-specific metrics
        code_metrics = {
            "mean_solution_length": (
                np.mean([len(sol) for sol in processed_solutions])
                if processed_solutions
                else 0
            ),
            "solution_length_std": (
                np.std([len(sol) for sol in processed_solutions])
                if processed_solutions
                else 0
            ),
            "successful_ratio": correct_samples / n_samples if n_samples > 0 else 0,
        }

        result = {
            "problem_id": idx,
            "pass_at_k": pass_at_k,
            "error_stats": error_tracker.get_problem_stats(idx),
            "semantic_metrics": {**semantic_metrics, **code_metrics},
        }
        # Convert numpy types to native Python types before returning
        return convert_to_native_types(result)

    except Exception as e:
        logging.error(f"Error in problem evaluation: {str(e)}")
        return None


def calculate_aggregate_metrics(results):
    """Helper function to calculate aggregate metrics across all problems"""
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
    }


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(
        obj, (np.int64, np.int32)
    ):  # Add explicit handling for numpy integer types
        return int(obj)
    elif isinstance(
        obj, (np.float64, np.float32)
    ):  # Add explicit handling for numpy float types
        return float(obj)
    return obj


def calculate_semantic_metrics(
    processed_solutions, canonical_solution, solution_log_probs, entailment_model
):
    """Calculate semantic metrics for code solutions"""
    if not processed_solutions:
        return {}

    try:
        # Extract implementation bodies
        solution_bodies = []
        for sol in processed_solutions:
            implementation = extract_function_body(sol)
            if implementation:
                solution_bodies.append(implementation)

        if not solution_bodies:
            return {}

        # Calculate semantic clustering
        semantic_ids = get_semantic_ids(solution_bodies, entailment_model)
        semantic_cluster_counts = np.bincount(semantic_ids)

        # Calculate entailment scores
        canonical_alignments = []
        reverse_alignments = []

        for solution in solution_bodies:
            canon_align = context_entails_response(
                canonical_solution, [solution], entailment_model
            )
            canonical_alignments.append(canon_align)

            rev_align = context_entails_response(
                solution, [canonical_solution], entailment_model
            )
            reverse_alignments.append(rev_align)

        # Calculate average alignments
        canonical_alignment = statistics.mean(canonical_alignments)
        reverse_alignment = statistics.mean(reverse_alignments)
        bidirectional = (canonical_alignment + reverse_alignment) / 2

        return {
            "semantic_entropy": cluster_assignment_entropy(semantic_ids),
            "predictive_entropy": predictive_entropy(solution_log_probs),
            "predictive_entropy_rao": predictive_entropy_rao(solution_log_probs),
            "num_semantic_clusters": len(set(semantic_ids)),
            "largest_cluster_size": max(semantic_cluster_counts),
            "cluster_size_std": np.std(semantic_cluster_counts),
            "canonical_alignment": canonical_alignment,
            "reverse_alignment": reverse_alignment,
            "bidirectional_alignment": bidirectional,
            "semantic_diversity": len(set(semantic_ids)) / len(semantic_ids),
            "majority_solution_frequency": max(semantic_cluster_counts)
            / len(semantic_ids),
        }

    except Exception as e:
        logging.error(f"Error calculating semantic metrics: {str(e)}")
        return {}


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
    experiment_dir = create_experiment_dir()
    log_filename = setup_logging(experiment_dir)
    # Model parameters - Using a smaller model for the approximation
    target_model_name = "meta-llama/Llama-3.2-3B"
    approx_model_name = "meta-llama/Llama-3.2-1b"  # Smaller model for draft

    # Load dataset
    logging.info("Loading dataset...")
    dataset = get_dataset("openai_humaneval", seed=42)

    # Initialize the speculative sampling model
    logging.info("Initializing speculative sampling model...")
    approx_model, target_model, tokenizer = load_approx_and_target_model_and_tokenizer(
        approx_model_name=approx_model_name,
        target_model_name=target_model_name,
    )

    # Load entailment model
    logging.info("Loading entailment model...")
    entailment_model = EntailmentDeberta()

    # Evaluate
    logging.info("Starting evaluation...")

    aggregate_metrics, detailed_results, error_stats = evaluate_model(
        approx_model,
        target_model,
        tokenizer,
        dataset,
        num_problems=164,
        n_samples=10,
        k=5,
        entailment_model=entailment_model,
        experiment_dir=experiment_dir,
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
        "target_model": target_model_name,
        "approx_model": approx_model_name,
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
