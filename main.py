import torch
from data import get_dataset
from model import load_model_and_tokenizer
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
import logging
import gc


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


def evaluate_model(
    model, tokenizer, dataset, num_problems, n_samples, k, entailment_model
):
    """
    Evaluate the model on the dataset with error tracking and semantic uncertainty metrics.
    """
    results = []
    error_tracker = ErrorTracker()
    device = next(model.parameters()).device

    for idx in tqdm(range(num_problems)):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"\n{'='*50}")
        logging.info(f"Problem {idx}")

        problem = dataset[idx]
        question = problem["question"]
        canonical_solution = problem["canonical_solution"]
        entry_point = problem["entry_point"]
        test_code = problem["test_code"]
        correct_samples = 0

        # Store all solutions (both raw and processed) and their scores
        raw_solutions = []
        processed_solutions = []
        solution_log_probs = []

        try:
            encoded_input = tokenizer(question, return_tensors="pt", truncation=True)
            input_ids = encoded_input["input_ids"].to(device)
            attention_mask = encoded_input.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.8,
                top_k=100,
                output_scores=True,
                num_return_sequences=n_samples,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=False,
                return_legacy_cache=False,
            )

            if hasattr(outputs, "scores") and outputs.scores:
                scores = outputs.scores
                for batch_idx in range(len(outputs.sequences)):
                    error_tracker.increment_total(idx)
                    generated_ids = outputs.sequences[batch_idx]
                    # Get indices of non-padding tokens
                    non_pad_indices = (
                        (generated_ids != tokenizer.pad_token_id).nonzero().squeeze(-1)
                    )
                    if len(non_pad_indices) > 0:
                        start_idx = non_pad_indices[0].item()

                        for step, score in enumerate(scores):
                            if isinstance(score, tuple):
                                score = score[0]
                            step_log_probs = torch.log_softmax(score, dim=-1)

                            # Only include if we're past the prompt
                            if step + start_idx + 1 < len(generated_ids):
                                token = generated_ids[step + start_idx + 1]

                                # Skip padding tokens
                                if token == tokenizer.pad_token_id:
                                    continue

                                # Get probability for this specific sequence's token
                                log_prob_step = step_log_probs[batch_idx, token].item()

                                # Weight important tokens more heavily
                                if token in [
                                    tokenizer.convert_tokens_to_ids(t)
                                    for t in ["return", "while", "if", "for"]
                                ]:
                                    log_prob_step *= (
                                        1.2  # Boost probability for structural tokens
                                    )

                                if not np.isfinite(log_prob_step):
                                    log_prob_step = -10.0

                                log_prob += log_prob_step
                                sequence_length += 1

                        if sequence_length > 0:
                            log_prob = log_prob / sequence_length
                            # Remove this scaling factor as it's reducing the differences
                            # log_prob = log_prob / 5.0

                        # Use a wider range for clipping
                        log_prob = np.clip(log_prob, -10.0, 0.0)
                    else:
                        log_prob = 0.0

                    solution_log_probs.append(log_prob)

                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    raw_solutions.append(response)
                    

                    # Extract and fix the function
                    generated_code = ""
                    if "def " + entry_point in response:
                        start = response.find("def " + entry_point)
                        generated_code = response[start:]
                        generated_code = extract_and_fix_function(
                            generated_code, entry_point
                        )

                        if generated_code:
                            processed_solutions.append(generated_code)
                            logging.info(
                                f"Generated code after MAIN extraction: {generated_code}"
                            )

                    # Run tests
                    test_env = create_test_env()
                    if try_run_tests(response, entry_point, test_code, test_env):
                        correct_samples += 1
                        logging.info("✓ Sample passed all tests on raw response")
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
                            logging.info("✓ Sample passed all tests after fixing")
                            continue

                    logging.info("✗ Sample failed all test attempts")

        except Exception as e:
            error_tracker.add_error(idx, type(e).__name__)
            logging.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
            continue

        # Calculate semantic metrics using the best available solutions
        solutions_for_metrics = (
            processed_solutions if processed_solutions else raw_solutions
        )
        semantic_metrics = {}

        if solutions_for_metrics:
            logging.info(
                f"\nCalculating semantic metrics for {len(solutions_for_metrics)} solutions"
            )
            logging.debug(
                "Sample solution lengths: "
                + str([len(sol) for sol in solutions_for_metrics[:3]])
                + "..."
            )

            # Calculate entropy metrics
            semantic_ids = get_semantic_ids(solutions_for_metrics, entailment_model)
            num_clusters = len(set(semantic_ids))
            logging.info(f"Number of semantic clusters: {num_clusters}")

            semantic_entropy = cluster_assignment_entropy(semantic_ids)
            logging.info(f"Semantic entropy: {semantic_entropy:.3f}")

            if solution_log_probs:
                logging.info(f"Log probabilities: {solution_log_probs}")
                pred_entropy = predictive_entropy(solution_log_probs)
                pred_entropy_rao = predictive_entropy_rao(solution_log_probs)
                logging.info(f"Predictive entropy: {pred_entropy:.3f}")
                logging.info(f"Predictive entropy Rao: {pred_entropy_rao:.3f}")
            else:
                pred_entropy = pred_entropy_rao = 0.0

            # Calculate alignments
            canonical_alignments = []
            reverse_alignments = []

            for solution in solutions_for_metrics:
                canon_align = context_entails_response(
                    canonical_solution, [solution], entailment_model
                )
                canonical_alignments.append(canon_align)

                rev_align = context_entails_response(
                    solution, [canonical_solution], entailment_model
                )
                reverse_alignments.append(rev_align)

            canonical_alignment = (
                sum(canonical_alignments) / len(canonical_alignments)
                if canonical_alignments
                else 0.0
            )
            reverse_alignment = (
                sum(reverse_alignments) / len(reverse_alignments)
                if reverse_alignments
                else 0.0
            )
            bidirectional = (canonical_alignment + reverse_alignment) / 2

            semantic_metrics = {
                "semantic_entropy": semantic_entropy,
                "predictive_entropy": pred_entropy,
                "predictive_entropy_rao": pred_entropy_rao,
                "num_semantic_clusters": num_clusters,
                "num_solutions": len(solutions_for_metrics),
                "num_processed_solutions": len(processed_solutions),
                "canonical_alignment": canonical_alignment,
                "reverse_alignment": reverse_alignment,
                "bidirectional_alignment": bidirectional,
            }

        # Record results
        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)
        results.append(
            {
                "problem_id": idx,
                "pass_at_k": pass_at_k,
                "error_stats": error_tracker.get_problem_stats(idx),
                "semantic_metrics": semantic_metrics,
            }
        )

        # Log results
        logging.info(f"Problem {idx} Results:")
        logging.info(f"pass@{k}: {pass_at_k:.2f}")
        if semantic_metrics:
            logging.info(f"Semantic metrics: {semantic_metrics}")
            logging.info(
                f"Canonical solution alignment: {semantic_metrics['canonical_alignment']:.2f}"
            )
            logging.info(
                f"Bidirectional alignment: {semantic_metrics['bidirectional_alignment']:.2f}"
            )

    # Calculate aggregate metrics
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


def calculate_implementation_log_prob(
    implementation: str,
    generated_ids: torch.Tensor,
    scores: List[torch.Tensor],
    tokenizer,
    batch_idx: int,
) -> float:
    """Calculate average log probability for implementation tokens.

    Args:
        implementation: The extracted implementation code
        full_response: Full model response
        generated_ids: Token ids generated by the model
        scores: Log probability scores for each generation step
        tokenizer: Tokenizer used for encoding
        batch_idx: Index of the current batch sample

    Returns:
        float: Average log probability across implementation tokens
    """
    # Encode just the implementation
    impl_tokens = tokenizer.encode(implementation, add_special_tokens=False)

    # Get non-padding tokens
    non_pad_indices = (generated_ids != tokenizer.pad_token_id).nonzero().squeeze(-1)
    if len(non_pad_indices) == 0:
        return float("-inf")

    start_idx = non_pad_indices[0].item()

    # Calculate log probs for each implementation token
    log_probs = []

    for token in impl_tokens:
        # Find this token in the generated sequence
        for step, score in enumerate(scores):
            if isinstance(score, tuple):
                score = score[0]

            token_idx = step + start_idx + 1
            if token_idx >= len(generated_ids):
                continue

            if generated_ids[token_idx].item() == token:
                # Calculate log prob for this token
                step_log_probs = torch.log_softmax(score, dim=-1)
                log_prob = step_log_probs[batch_idx, token].item()
                log_probs.append(log_prob)
                break
    
    return sum(log_probs)


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
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # need to add HF_TOKEN

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
        num_problems=1,
        n_samples=5,
        k=2,
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
