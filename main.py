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
    generate_branching_responses,
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
                'problem_id': result['problem_id'],
                'pass_at_k': result['pass_at_k']
            }
            # Add semantic metrics if they exist
            if result['semantic_metrics']:
                clean_result.update(result['semantic_metrics'])
            # Add error stats
            for error_type, count in result['error_stats'].items():
                if error_type != 'total_samples':
                    clean_result[f'error_{error_type}'] = count
            cleaned_results.append(clean_result)
            
        self.results = pd.DataFrame(cleaned_results)
        self.experiment_dir = experiment_dir

    def plot_metric_distribution(self, metric_name, title=None):
        """Create a distribution plot for a specific metric"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.results[metric_name], kde=True)
        plt.title(title or f"Distribution of {metric_name}")
        plt.xlabel(metric_name)
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.experiment_dir, f"{metric_name}_distribution.png"))
        plt.close()

    def plot_metric_correlations(self):
        """Create a correlation heatmap between different metrics"""
        plt.figure(figsize=(12, 10))
        numeric_cols = self.results.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.results[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Between Code Generation Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "metric_correlations.png"))
        plt.close()

    def plot_metrics_over_problems(self):
        """Plot all metrics across problems"""
        metrics = [col for col in self.results.columns if col != 'problem_id']
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(range(len(self.results)), 
                    self.results[metric], 
                    label=metric, 
                    marker='o')
        plt.title("Metrics across Problems")
        plt.xlabel("Problem Index")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "metrics_across_problems.png"))
        plt.close()

    def plot_error_distributions(self):
        """Plot distribution of different error types"""
        error_columns = [col for col in self.results.columns if col.startswith('error_')]
        if error_columns:
            plt.figure(figsize=(10, 6))
            error_data = self.results[error_columns].sum()
            error_data.plot(kind='bar')
            plt.title("Distribution of Error Types")
            plt.xlabel("Error Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, "error_distributions.png"))
            plt.close()

    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        summary_stats = self.results.describe()
        summary_stats.to_csv(os.path.join(self.experiment_dir, "summary_statistics.csv"))
        return summary_stats

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

def setup_logging(experiment_dir):
    """Configure logging with detailed formatting and both file and console handlers"""
    log_filename = os.path.join(experiment_dir, "humaneval.log")
    
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )

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

def evaluate_model(
    model, 
    tokenizer, 
    dataset, 
    num_problems, 
    n_samples, 
    k, 
    entailment_model,
    experiment_dir
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
            model,
            tokenizer,
            problem,
            idx,
            n_samples,
            k,
            entailment_model,
            error_tracker
        )
        
        if metrics:
            results.append(metrics)
            
            # Log metrics
            logging.info(f"\nProblem {idx} Results:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{key}: {value:.4f}")

    # Generate visualizations
    visualizer = ResultsVisualizer(results, experiment_dir)
    try:
        visualizer.plot_metric_correlations()
        visualizer.plot_metrics_over_problems()
        visualizer.plot_error_distributions()
        summary_stats = visualizer.generate_summary_statistics()
        logging.info(f"Generated summary statistics:\n{summary_stats}")

        # Save detailed results
        results_file = os.path.join(experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")

    # Calculate and return aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)
    return aggregate_metrics, results, error_tracker.get_total_stats()

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
    return obj

def evaluate_problem(
    model,
    tokenizer,
    problem,
    idx,
    n_samples,
    k,
    entailment_model,
    error_tracker
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
        # Generate solutions using branching method
        responses = generate_branching_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            max_length=256,
            num_branches=n_samples
        )

        # Process each generated response
        for response, confidence_score, log_prob in responses:
            error_tracker.increment_total(idx)
            
            raw_solutions.append(response)
            scaled_log_prob = np.clip(log_prob, -10.0, 0.0)
            solution_log_probs.append(scaled_log_prob)

            # Extract and process function
            if "def " + entry_point in response:
                generated_code = extract_and_fix_function(
                    response[response.find("def " + entry_point):],
                    entry_point
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
                    idx
                ):
                    correct_samples += 1
                    continue

        # Calculate semantic metrics
        semantic_metrics = calculate_semantic_metrics(
            processed_solutions,
            canonical_solution,
            solution_log_probs,
            entailment_model
        )

        # Calculate pass@k
        pass_at_k = calculate_pass_at_k(n_samples, correct_samples, k)

        # Additional code-specific metrics
        code_metrics = {
            'mean_solution_length': np.mean([len(sol) for sol in processed_solutions]) if processed_solutions else 0,
            'solution_length_std': np.std([len(sol) for sol in processed_solutions]) if processed_solutions else 0,
            'successful_ratio': correct_samples / n_samples if n_samples > 0 else 0,
        }

        result = {
            'problem_id': idx,
            'pass_at_k': pass_at_k,
            'error_stats': error_tracker.get_problem_stats(idx),
            'semantic_metrics': {**semantic_metrics, **code_metrics}
        }
        # Convert numpy types to native Python types before returning
        return convert_to_native_types(result)

    except Exception as e:
        logging.error(f"Error in problem evaluation: {str(e)}")
        return None
    
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


def calculate_semantic_metrics(
    processed_solutions,
    canonical_solution,
    solution_log_probs,
    entailment_model
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
                canonical_solution,
                [solution],
                entailment_model
            )
            canonical_alignments.append(canon_align)

            rev_align = context_entails_response(
                solution,
                [canonical_solution],
                entailment_model
            )
            reverse_alignments.append(rev_align)

        # Calculate average alignments
        canonical_alignment = statistics.mean(canonical_alignments)
        reverse_alignment = statistics.mean(reverse_alignments)
        bidirectional = (canonical_alignment + reverse_alignment) / 2

        return {
            'semantic_entropy': cluster_assignment_entropy(semantic_ids),
            'predictive_entropy': predictive_entropy(solution_log_probs),
            'predictive_entropy_rao': predictive_entropy_rao(solution_log_probs),
            'num_semantic_clusters': len(set(semantic_ids)),
            'largest_cluster_size': max(semantic_cluster_counts),
            'cluster_size_std': np.std(semantic_cluster_counts),
            'canonical_alignment': canonical_alignment,
            'reverse_alignment': reverse_alignment,
            'bidirectional_alignment': bidirectional,
            'semantic_diversity': len(set(semantic_ids)) / len(semantic_ids),
            'majority_solution_frequency': max(semantic_cluster_counts) / len(semantic_ids)
        }

    except Exception as e:
        logging.error(f"Error calculating semantic metrics: {str(e)}")
        return {}

def main():
    # Create experiment directory and setup logging
    experiment_dir = create_experiment_dir()
    log_filename = setup_logging(experiment_dir)
    logging.info("Starting enhanced HumanEval evaluation")

    try:
        # Model parameters
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

        # Load dataset
        logging.info("Loading dataset...")
        dataset = get_dataset("openai_humaneval", seed=42)

        # Load models
        logging.info("Loading models...")
        model, tokenizer = load_model_and_tokenizer(model_name)
        entailment_model = CodeAwareDeberta()

        # Run evaluation
        logging.info("Starting evaluation...")
        aggregate_metrics, detailed_results, error_stats = evaluate_model(
            model,
            tokenizer,
            dataset,
            num_problems=3,
            n_samples=10,
            k=2,
            entailment_model=entailment_model,
            experiment_dir=experiment_dir
        )

        # Save final results
        results = {
            'model_name': model_name,
            'aggregate_metrics': convert_to_native_types(aggregate_metrics),
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(dataset),
            'error_statistics': convert_to_native_types(error_stats),
            'detailed_results': convert_to_native_types(detailed_results)
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
        logging.info(f"Mean semantic entropy: {aggregate_metrics['mean_semantic_entropy']:.2f}")
        logging.info(f"Mean predictive entropy: {aggregate_metrics['mean_predictive_entropy']:.2f}")
        logging.info(f"Mean canonical alignment: {aggregate_metrics['mean_canonical_alignment']:.2f}")
        logging.info(f"\nError Statistics:")
        logging.info(json.dumps(error_stats, indent=2))

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()