import os
import sys
from datetime import datetime
import logging
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List, Optional, Union, Dict, Tuple, Any
import statistics
import math
import gc
import contextlib
import io
import timeout_decorator
import ast
from model import (
    CodeAwareDeberta,
    load_model_and_tokenizer,
)
from data import get_dataset
from scores import (
    get_semantic_ids,
    cluster_assignment_entropy,
    predictive_entropy,
    predictive_entropy_rao,
    context_entails_response,
)

# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_experiment_dir():
    """Create a timestamped directory for the current experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"humaneval_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


class ResultsVisualizer:
    def __init__(self, results_list, experiment_dir):
        """Initialize with a list of result dictionaries and experiment directory"""
        # Clean results by removing non-numeric and nested data
        cleaned_results = []
        for result in results_list:
            clean_result = {
                "problem_id": result["problem_id"],
                "pass_at_k": result["pass_at_k"],
            }
            # Add semantic metrics if they exist
            if result["semantic_metrics"]:
                clean_result.update(result["semantic_metrics"])
            # Add error stats
            for error_type, count in result["error_stats"].items():
                if error_type != "total_samples":
                    clean_result[f"error_{error_type}"] = count
            cleaned_results.append(clean_result)

        self.results = pd.DataFrame(cleaned_results)
        self.experiment_dir = experiment_dir

    def plot_metrics_over_problems(self):
        """Plot all metrics across problems"""
        metrics = [col for col in self.results.columns if col != "problem_id"]
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(
                range(len(self.results)), self.results[metric], label=metric, marker="o"
            )
        plt.title("Metrics across Problems")
        plt.xlabel("Problem Index")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "metrics_across_problems.png"))
        plt.close()

    def plot_alignment_triangle(self):
        """Create triangular visualization for alignment relationships"""
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.results["canonical_alignment"],
            self.results["reverse_alignment"],
            c=self.results["bidirectional_alignment"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(label="Bidirectional Alignment Score")
        max_val = max(
            self.results["canonical_alignment"].max(),
            self.results["reverse_alignment"].max(),
        )
        plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect Balance")
        plt.xlabel("Canonical Alignment")
        plt.ylabel("Reverse Alignment")
        plt.title("Alignment Triangle Visualization")
        plt.legend()
        plt.savefig(os.path.join(self.experiment_dir, "alignment_triangle.png"))
        plt.close()

    def plot_entropy_landscape(self):
        """Create 2D entropy landscape visualization"""
        plt.figure(figsize=(12, 8))
        self.results["cluster_entropy"] = -np.log2(
            self.results["largest_cluster_size"] / self.results["num_semantic_clusters"]
        )

        ax = plt.axes(projection="3d")
        scatter = ax.scatter(
            self.results["semantic_entropy"],
            self.results["predictive_entropy"],
            self.results["cluster_entropy"],
            c=self.results["pass_at_k"],
            cmap="coolwarm",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Pass@k Score")
        ax.set_xlabel("Semantic Entropy")
        ax.set_ylabel("Predictive Entropy")
        ax.set_zlabel("Cluster Entropy")
        plt.title("Entropy Landscape")
        plt.savefig(os.path.join(self.experiment_dir, "entropy_landscape.png"))
        plt.close()

    def plot_error_distributions(self):
        """Plot distribution of different error types"""
        error_columns = [
            col for col in self.results.columns if col.startswith("error_")
        ]
        if error_columns:
            plt.figure(figsize=(10, 6))
            error_data = self.results[error_columns].sum()
            error_data.plot(kind="bar")
            plt.title("Distribution of Error Types")
            plt.xlabel("Error Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, "error_distributions.png"))
            plt.close()

    def plot_solution_quality_matrix(self):
        """Create correlation matrix for solution quality metrics"""
        metrics = [
            "semantic_entropy",
            "predictive_entropy",
            "canonical_alignment",
            "bidirectional_alignment",
            "pass_at_k",
        ]

        corr_matrix = self.results[metrics].corr()
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
        )
        plt.title("Solution Quality Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "solution_quality_matrix.png"))
        plt.close()

    def generate_semantic_diversity_report(self):
        """Generate detailed report on semantic diversity metrics"""
        try:
            metrics = {
                "semantic_clusters_stats": {
                    "mean": float(self.results["num_semantic_clusters"].mean()),
                    "std": float(self.results["num_semantic_clusters"].std()),
                    "max": int(self.results["num_semantic_clusters"].max()),
                    "min": int(self.results["num_semantic_clusters"].min()),
                },
                "diversity_vs_performance": float(
                    np.corrcoef(
                        self.results["semantic_entropy"], self.results["pass_at_k"]
                    )[0, 1]
                ),
                "entropy_correlations": {
                    "semantic_vs_predictive": float(
                        np.corrcoef(
                            self.results["semantic_entropy"],
                            self.results["predictive_entropy"],
                        )[0, 1]
                    )
                },
            }

            report_file = os.path.join(
                self.experiment_dir, "semantic_diversity_report.json"
            )
            with open(report_file, "w") as f:
                json.dump(metrics, f, indent=2)

            return metrics

        except Exception as e:
            logging.error(f"Error generating semantic diversity report: {str(e)}")
            return {}


def setup_logging(experiment_dir):
    """Configure logging with detailed formatting"""
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


def evaluate_model(
    model,
    tokenizer,
    dataset,
    num_problems,
    n_samples,
    k,
    entailment_model,
    experiment_dir,
):
    """Enhanced evaluation function with visualization and metrics"""
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

        raw_solutions = []
        processed_solutions = []
        solution_log_probs = []

        try:
            # Generate solutions
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

            # Process outputs and calculate metrics
            if hasattr(outputs, "scores") and outputs.scores:
                for batch_idx in range(len(outputs.sequences)):
                    error_tracker.increment_total(idx)
                    generated_ids = outputs.sequences[batch_idx]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    raw_solutions.append(response)

                    # Calculate log probabilities
                    log_prob = calculate_sequence_log_prob(
                        generated_ids, outputs.scores, tokenizer, batch_idx
                    )
                    solution_log_probs.append(log_prob)

                    # Process and test solutions
                    if process_and_test_solution(
                        response, entry_point, test_code, error_tracker, idx
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

            results.append(
                {
                    "problem_id": idx,
                    "pass_at_k": pass_at_k,
                    "error_stats": error_tracker.get_problem_stats(idx),
                    "semantic_metrics": semantic_metrics,
                }
            )

        except Exception as e:
            error_tracker.add_error(idx, type(e).__name__)
            logging.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
            continue

    # Generate visualizations
    visualizer = ResultsVisualizer(results, experiment_dir)
    visualizer.plot_metrics_over_problems()
    visualizer.plot_error_distributions()
    visualizer.plot_alignment_triangle()
    visualizer.plot_entropy_landscape()
    visualizer.plot_solution_quality_matrix()
    diversity_metrics = visualizer.generate_semantic_diversity_report()

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)
    return aggregate_metrics, results, error_tracker.get_total_stats()


def main():
    # Create experiment directory and setup logging
    experiment_dir = create_experiment_dir()
    log_filename = setup_logging(experiment_dir)
    logging.info("Starting enhanced HumanEval evaluation")

    try:
        # Model parameters
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

        # Load dataset and models
        logging.info("Loading dataset...")
        dataset = get_dataset("openai_humaneval", seed=42)

        logging.info("Loading models...")
        model, tokenizer = load_model_and_tokenizer(model_name)
        entailment_model = CodeAwareDeberta()

        # Run evaluation
        aggregate_metrics, detailed_results, error_stats = evaluate_model(
            model,
            tokenizer,
            dataset,
            num_problems=164,
            n_samples=5,
            k=2,
            entailment_model=entailment_model,
            experiment_dir=experiment_dir,
        )

        # Save results
        results = {
            "model_name": model_name,
            "aggregate_metrics": aggregate_metrics,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(dataset),
            "error_statistics": error_stats,
            "detailed_results": detailed_results,
        }

        results_file = os.path.join(experiment_dir, "final_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"\nFinal Results:")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Experiment directory: {experiment_dir}")
        logging.info(f"Log file: {log_filename}")

        # Print key metrics
        logging.info("\nKey Metrics:")
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
        logging.info(f"\nError Statistics:")
        logging.info(json.dumps(error_stats, indent=2))

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


def calculate_sequence_log_prob(generated_ids, scores, tokenizer, batch_idx):
    """Calculate log probability for a generated sequence"""
    log_prob = 0
    sequence_length = 0

    non_pad_indices = (generated_ids != tokenizer.pad_token_id).nonzero().squeeze(-1)
    if len(non_pad_indices) > 0:
        start_idx = non_pad_indices[0].item()

        for step, score in enumerate(scores):
            if isinstance(score, tuple):
                score = score[0]
            step_log_probs = torch.log_softmax(score, dim=-1)

            if step + start_idx + 1 < len(generated_ids):
                token = generated_ids[step + start_idx + 1]

                if token == tokenizer.pad_token_id:
                    continue

                log_prob_step = step_log_probs[batch_idx, token].item()

                if token in [
                    tokenizer.convert_tokens_to_ids(t)
                    for t in ["return", "while", "if", "for"]
                ]:
                    log_prob_step *= 1.2

                if not np.isfinite(log_prob_step):
                    log_prob_step = -10.0

                log_prob += log_prob_step
                sequence_length += 1

        if sequence_length > 0:
            log_prob = log_prob / sequence_length
            log_prob = np.clip(log_prob, -10.0, 0.0)

    return log_prob


def process_and_test_solution(response, entry_point, test_code, error_tracker, idx):
    """Process and test a generated solution"""
    # Try running tests on raw response
    test_env = create_test_env()
    if try_run_tests(response, entry_point, test_code, test_env):
        return True

    # Extract and fix function if needed
    generated_code = ""
    if "def " + entry_point in response:
        start = response.find("def " + entry_point)
        generated_code = response[start:]
        generated_code = extract_and_fix_function(generated_code, entry_point)

        if generated_code:
            test_env = create_test_env()
            if try_run_tests(
                generated_code, entry_point, test_code, test_env, error_tracker, idx
            ):
                return True

    return False


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

        # Calculate additional metrics
        semantic_diversity = len(set(semantic_ids)) / len(semantic_ids)
        majority_solution_freq = max(semantic_cluster_counts) / len(semantic_ids)

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
            "semantic_diversity": semantic_diversity,
            "majority_solution_frequency": majority_solution_freq,
            "mean_solution_length": np.mean([len(sol) for sol in solution_bodies]),
            "solution_length_std": np.std([len(sol) for sol in solution_bodies]),
        }

    except Exception as e:
        logging.error(f"Error calculating semantic metrics: {str(e)}")
        return {}


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
