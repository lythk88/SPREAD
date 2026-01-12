#!/usr/bin/env python3
"""
Comprehensive evaluation script for comparing all generation strategies.

This script evaluates the results of various generation strategies including:
- Greedy search
- Temperature sampling
- Beam search
- Nucleus sampling
- Diverse beam search
- Beam search multinomial sampling
- Contrastive search
- Speculative decoding
- Shapley-DPP Adaptive Decoding

It calculates pass@k metrics, token efficiency, runtime performance, and more.
"""

import os
import re
import sys
import glob
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict

# Add parent directory to path to import properly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.parser import extract_answer, strip_string, parse_ground_truth
from src.evaluation.grader import math_equal_process

def parse_args():
    """Parse command-line arguments for comprehensive evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate and compare all generation strategies")
    
    # Input options
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing result files for all strategies"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (defaults to input_dir/evaluation)"
    )
    
    # Strategy selection
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated list of strategies to evaluate, or 'all' for all available"
    )
    
    # Evaluation options
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated list of k values for pass@k evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="olympiadbench",
        choices=["math", "aime24", "olympiadbench"],
        help="Dataset being evaluated"
    )
    
    # Output options
    parser.add_argument(
        "--save_table",
        action="store_true",
        help="Save evaluation table to CSV file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory for plots (defaults to output_dir/plots)"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None, 
        help="Custom name for the run (default: auto-generated)"
    )
    
    return parser.parse_args()

def extract_candidate_answer(candidate: Dict[str, Any], dataset: str = "olympiadbench") -> str:
    """
    Extract the answer from a candidate solution.

    Args:
        candidate: Candidate solution dictionary
        dataset: Dataset name for extraction logic

    Returns:
        Extracted answer string
    """
    text = candidate.get("text", "")
    
    # If candidate is a string, use it directly
    if isinstance(candidate, str):
        text = candidate
    # Handle result format from various decoding strategies
    elif isinstance(candidate, dict):
        # Try different field names that might contain the text
        for field in ["text", "response", "answer", "generation"]:
            if field in candidate:
                text = candidate[field]
                if isinstance(text, str):
                    break
    
    try:
        # First try to extract with the standard function
        answer = extract_answer(text, data_name=dataset)
        if answer and answer.strip():
            return strip_string(answer)

        if "Answer:" in text or "answer:" in text or "answer is:" in text:
            # Find answer after "Answer:" tag
            if "Answer:" in text:
                parts = text.split("Answer:")
            else:
                parts = text.split("answer:")

            answer_text = parts[-1].strip()

            # Get the first sentence of the answer
            if "." in answer_text:
                answer_text = answer_text.split(".")[0].strip()

            # Take only the final number if it seems to be a calculated result
            number_match = re.search(r'-?\d*\.?\d+', answer_text)
            if number_match:
                return number_match.group(0)

            return strip_string(answer_text)

        # Try to find the last number in the text as a last resort
        number_matches = re.findall(r'-?\d*\.?\d+', text)
        if number_matches:
            return number_matches[-1]

        return ""
    except Exception as e:
        print(f"Error extracting answer: {e}")
        # Last resort fallback
        return text.strip()

def evaluate_candidate(candidate: Dict[str, Any], ground_truth: str, dataset: str = "olympiadbench") -> bool:
    """
    Evaluate if a candidate answer is correct.

    Args:
        candidate: Candidate solution
        ground_truth: Ground truth answer
        dataset: Dataset name

    Returns:
        True if the candidate is correct, False otherwise
    """
    # Extract answer from candidate
    answer = extract_candidate_answer(candidate, dataset)

    if not answer:
        return False

    try:
        # Parse ground truth
        try:
            _, gt = parse_ground_truth(
                {"problem": "", "answer": ground_truth, "solution": ""},
                data_name=dataset
            )
            # If parse_ground_truth returns empty string, use the original ground_truth
            if not gt or gt.strip() == '':
                print(f"Warning: parse_ground_truth returned empty string, using original ground truth")
                gt = ground_truth
        except Exception as e:
            print(f"Warning: Error parsing ground truth: {e}")
            gt = ground_truth

        # Try direct string comparison first
        if answer.strip() == gt.strip():
            return True

        # Try with math_equal_process
        try:
            result = math_equal_process((0, answer, gt))
            if result:
                return True
        except Exception as e:
            print(f"Warning: Error in math_equal_process: {e}")

        # Try numeric comparison for numbers
        try:
            # Check if both are numeric values
            answer_val = float(answer.replace(',', ''))
            gt_val = float(gt.replace(',', ''))

            # Check if they're close enough
            return abs(answer_val - gt_val) < 1e-6 or abs(answer_val - gt_val) / max(abs(gt_val), 1e-10) < 1e-4
        except ValueError:
            # Not numeric values
            pass

        # Try relaxed string comparison
        answer_clean = answer.lower().strip().replace(' ', '').replace(',', '')
        gt_clean = gt.lower().strip().replace(' ', '').replace(',', '')
        if answer_clean == gt_clean:
            return True

        return False
    except Exception as e:
        print(f"Error evaluating answer: {e}")

        # Last resort: try direct numeric comparison
        try:
            # Try to extract numbers from both strings
            import re
            answer_nums = re.findall(r'-?\d*\.?\d+', answer)
            gt_nums = re.findall(r'-?\d*\.?\d+', ground_truth)

            if answer_nums and gt_nums:
                return answer_nums[-1] == gt_nums[-1]
        except:
            pass

        return False

def calculate_empirical_pass_at_k(candidates: List[Dict[str, Any]], 
                                 ground_truth: str, 
                                 k: int, 
                                 dataset: str = "olympiadbench") -> float:
    """
    Calculate empirical pass@k by checking if any of the top k candidates have the correct answer.
    
    Args:
        candidates: List of candidate solutions
        ground_truth: Ground truth answer
        k: Value of k (number of candidates to consider)
        dataset: Dataset name
        
    Returns:
        1.0 if any of the top k candidates are correct, 0.0 otherwise
    """
    if not candidates:
        return 0.0
    
    # Take the top k candidates
    top_k = candidates[:min(k, len(candidates))]
    
    # Check if any are correct
    for candidate in top_k:
        if evaluate_candidate(candidate, ground_truth, dataset):
            return 1.0
    
    return 0.0

def load_result_file(file_path: str) -> Dict[str, Any]:
    """
    Load a result file in pickle format.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary with results data
    """
    print(f"Loading result file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        
            
        # Standardize structure for different file formats
        if "metadata" not in data:
            # Create metadata if it doesn't exist
            data["metadata"] = {"method": "unknown"}
            
        if "results" not in data:
            # If it's just a list of results
            if isinstance(data, list):
                data = {"metadata": {"method": "unknown"}, "results": data}
            else:
                # No clear format, assume the whole thing is one result
                data = {"metadata": {"method": "unknown"}, "results": [data]}
                
        # Try to infer method if unknown
        if data["metadata"].get("method", "unknown") == "unknown":
            # Try to infer from filename
            filename = os.path.basename(file_path)
            if "greedy" in filename:
                data["metadata"]["method"] = "greedy"
            elif "temperature" in filename or "temp" in filename:
                data["metadata"]["method"] = "temperature"
            elif "beam" in filename and "diverse" not in filename and "multinomial" not in filename:
                data["metadata"]["method"] = "beam"
            elif "nucleus" in filename or "topp" in filename:
                data["metadata"]["method"] = "nucleus"
            elif "diverse_beam" in filename:
                data["metadata"]["method"] = "diverse_beam"
            elif "beam_multinomial" in filename:
                data["metadata"]["method"] = "beam_multinomial"
            elif "contrastive" in filename:
                data["metadata"]["method"] = "contrastive"
            elif "speculative" in filename:
                data["metadata"]["method"] = "speculative"
            elif "sdad" in filename or "shapley_dpp" in filename:
                data["metadata"]["method"] = "shapley_dpp"
            elif "bestofn" in filename or "best-of" in filename:
                data["metadata"]["method"] = "best_of_n"
            elif "steer-algor1_" in filename or "steeraglor1" in filename:
                data["metadata"]["method"] = "steer-algor1_"
            elif "steer-base_" in filename or "steerbase" in filename:
                data["metadata"]["method"] = "steer-base_"
            elif "steer-algor1-ver2" in filename or "steeraglor1-ver2" in filename:
                data["metadata"]["method"] = "steer-algor1-ver2"
            elif "steer-base-ver2" in filename or "steerbase-ver2" in filename:
                data["metadata"]["method"] = "steer-base-ver2"
            elif "steer-algor1-ver3" in filename or "steeraglor1-ver3" in filename:
                data["metadata"]["method"] = "steer-algor1-ver3"
            elif "steer-base-ver3" in filename or "steerbase-ver3" in filename:
                data["metadata"]["method"] = "steer-base-ver3"
                
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return a minimal structure to avoid breaking the evaluation
        return {
            "metadata": {
                "method": "unknown",
                "error": str(e)
            },
            "results": []
        }

def find_strategy_files(input_dir: str) -> Dict[str, List[str]]:
    """
    Find result files for different strategies in the input directory.

    Args:
        input_dir: Directory to search for result files

    Returns:
        Dictionary mapping strategy names to lists of file paths
    """
    # Mapping of strategy search patterns to strategy names
    strategy_patterns = {
        "greedy": ["*greedy*raw.pkl", "*baseline_greedy*raw.pkl"],
        "temperature": ["*temperature*raw.pkl", "*baseline_temp*raw.pkl"],
        "beam": ["*baseline_beam*raw.pkl"],
        "nucleus": ["*nucleus*raw.pkl", "*topp*raw.pkl", "*baseline_nucleus*raw.pkl"],
        "diverse_beam": ["*diverse_beam*raw.pkl"],
        "beam_multinomial": ["*beam_multinomial*raw.pkl"],
        "contrastive": ["*contrastive*raw.pkl"],
        "speculative": ["*speculative*raw.pkl"],
        "shapley_dpp": ["*sdad*raw.pkl", "*shapley_dpp*raw.pkl"],
        "base_dpp": ["*base_dpp*raw.pkl"],
        "best_of_n": ["*bestofn*n*_raw.pkl"],  # Exact match for bestofn with specific pattern
        "nonshapley_dpp": ["*sdad*non-Shapley.pkl"],
        "steer-algor1_": ["*steer-algor1_*raw.pkl", "*steeraglor1_*raw.pkl"],
        "steer-base_": ["*steer-base_*raw.pkl", "*steerbase_*raw.pkl"],
        "steer-algor1-ver2": ["*steer-algor1-ver2*raw.pkl", "*steeraglor1-ver2*raw.pkl"],
        "steer-base-ver2": ["*steer-base-ver2*raw.pkl", "*steerbase-ver2*raw.pkl"],
        "steer-algor1-ver3": ["*steer-algor1-ver3*raw.pkl", "*steeraglor1-ver3*raw.pkl"],
        "steer-base-ver3": ["*steer-base-ver3*raw.pkl", "*steerbase-ver3*raw.pkl"],
        "steer": ["*steer*raw.pkl", "*steer*raw.pkl"],
        "baseline": ["*bestof*raw.pkl", "*bestof*raw.pkl"],
    }

    results = {}

    # Search for files matching each pattern
    for strategy, patterns in strategy_patterns.items():
        files = []
        for pattern in patterns:
            # Check in the root directory
            files.extend(glob.glob(os.path.join(input_dir, pattern)))
            # Also check in subdirectories
            files.extend(glob.glob(os.path.join(input_dir, "*", pattern)))

        if files:
            # Filter out duplicates
            files = sorted(list(set(files)))

            # Special handling for best_of_n to prevent duplication
            if strategy == "best_of_n":
                # Group by the N value in the filename (n16, n32, etc.)
                unique_n_values = {}
                for file_path in files:
                    # Extract N value from filename using regex
                    filename = os.path.basename(file_path)
                    import re
                    n_match = re.search(r'bestofn_n(\d+)_', filename)
                    if n_match:
                        n_value = n_match.group(1)
                        unique_key = f"best_of_{n_value}"
                        if unique_key not in unique_n_values:
                            unique_n_values[unique_key] = []
                        unique_n_values[unique_key].append(file_path)

                # Each N value gets its own entry in results
                for unique_key, unique_files in unique_n_values.items():
                    results[unique_key] = unique_files
                    print(f"Found {len(unique_files)} files for strategy '{unique_key}'")
            else:
                # Normal handling for other strategies
                results[strategy] = files
                print(f"Found {len(files)} files for strategy '{strategy}'")

    return results

def get_display_name(strategy: str, metadata: Dict[str, Any]) -> str:
    """
    Get a display name for a strategy with parameters.
    
    Args:
        strategy: Strategy name
        metadata: Metadata dictionary with parameters
        
    Returns:
        Display name with key parameters
    """
    if strategy == "greedy":
        return "Greedy"
    elif strategy == "temperature":
        temp = metadata.get("temperature", 1.0)
        return f"Temperature (T={temp})"
    elif strategy == "beam":
        beams = metadata.get("num_beams", "?")
        return f"Beam Search (b={beams})"
    elif strategy == "nucleus":
        top_p = metadata.get("top_p", 0.95)
        temp = metadata.get("temperature", 1.0)
        return f"Nucleus (p={top_p}, T={temp})"
    elif strategy == "diverse_beam":
        beams = metadata.get("num_beams", "?")
        groups = metadata.get("num_beam_groups", "?")
        penalty = metadata.get("diversity_penalty", "?")
        return f"Diverse Beam (b={beams}, g={groups}, p={penalty})"
    elif strategy == "beam_multinomial":
        beams = metadata.get("num_beams", "?")
        temp = metadata.get("temperature", 1.0)
        return f"Beam Multinomial (b={beams}, T={temp})"
    elif strategy == "contrastive":
        alpha = metadata.get("penalty_alpha", "?")
        top_k = metadata.get("top_k", "?")
        return f"Contrastive (α={alpha}, k={top_k})"
    elif strategy == "speculative":
        if metadata.get("do_sample", False):
            temp = metadata.get("temperature", 1.0)
            return f"Speculative (T={temp})"
        else:
            return "Speculative (greedy)"
    elif strategy == "shapley_dpp":
        return "Shapley-DPP"
    elif strategy == "base_dpp":
        return "Base-DPP"
    elif strategy == "best_of_n" or strategy.startswith("best_of_"):
        # Extract N value from strategy key for our custom naming (e.g., "best_of_32" -> "32")
        if strategy.startswith("best_of_") and len(strategy.split("_")) > 2:
            n_value = strategy.split("_")[-1]
        else:
            # If using the old naming scheme, try different ways to get the n value
            n_value = None
            
            # First check if it's in the metadata
            if "best_of" in metadata:
                n_value = metadata.get("best_of")
            # Check if it's in the display name already
            elif isinstance(display_name, str) and "best-of-" in display_name.lower():
                n_match = re.search(r'best-of-(\d+)', display_name.lower())
                if n_match:
                    n_value = n_match.group(1)
            
            # If still not found, use file path to extract n value
            if not n_value and "file_path" in metadata:
                file_path = metadata["file_path"]
                filename = os.path.basename(file_path)
                n_match = re.search(r'bestofn_n(\d+)_', filename)
                if n_match:
                    n_value = n_match.group(1)
            
            # Fallback
            if not n_value:
                n_value = "N"

        temp = metadata.get("temperature", 1.0)
        return f"Best-of-{n_value} (T={temp})"
    else:
        return strategy.replace("_", " ").title()

def evaluate_strategy(file_path: str, k_values: List[int], dataset: str) -> Dict[str, Any]:
    """
    Evaluate a strategy result file for pass@k and other metrics.
    
    Args:
        file_path: Path to the result file
        k_values: List of k values for pass@k evaluation
        dataset: Dataset name
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load the result file
    data = load_result_file(file_path)
    metadata = data["metadata"]
    # results = data["results"][:10]
    results = data["results"]
    
    # Get strategy name
    strategy = metadata.get("method", "unknown")
    
    # Store file_path in metadata for use in display name generation
    metadata["file_path"] = file_path
    
    # Basic metrics structure
    metrics = {
        "strategy": strategy,
        "display_name": get_display_name(strategy, metadata),
        "file_path": file_path,
        "metadata": metadata,
        "num_problems": len(results),
        "pass_at_k": {k: [] for k in k_values},
        "correct_examples": [],
        "incorrect_examples": []
    }
    
    # Process each problem
    for i, result in enumerate(tqdm(results, desc=f"Evaluating {strategy}")):
        # Extract problem, ground truth, and candidate(s)
        problem = result.get("problem", "")
        ground_truth = result.get("ground_truth", "")
        
        # Extract candidates - format varies by strategy
        candidates = []
        
        # Track if this result has majority voting
        has_majority_voting = "majority_voting" in result
        best_candidate_idx = 0

        # Get the majority-voted best candidate index if available
        # if has_majority_voting:
        #     voting_result = result["majority_voting"]
        #     best_candidate_idx = voting_result.get("best_candidate_idx", 0)

        # Handle different result formats
        # if "response" in result and "responses" in result:
            # Primary response + additional responses
            # candidates = [{"text": result["response"]}]
            # for resp in result.get("responses", []):
            #     if resp != result["response"]:  # Avoid duplicates
            #         candidates.append({"text": resp})
        if "responses" in result:
            # Multiple responses (common in sampling methods)
            for resp in result["responses"]:
                candidates.append({"text": resp})
        elif "response" in result:
            # Single response (common in greedy methods)
            if isinstance(result["response"], list):
                for resp in result["response"]:
                    candidates.append({"text": resp})
            else:
                candidates.append({"text": result["response"]})
        elif "candidates" in result:
            # Shapley-DPP format
            candidates = result["candidates"]
            
            # If we have majority voting, reorder candidates to put the best first
            # if has_majority_voting and 0 <= best_candidate_idx < len(candidates):
            #     # Move the best candidate to the front
            #     best_candidate = candidates[best_candidate_idx]
            #     candidates = [best_candidate] + [c for i, c in enumerate(candidates) if i != best_candidate_idx]
        elif "generations" in result:
            # Some methods use "generations"
            candidates = [{"text": g} for g in result["generations"]]
        elif "branches" in result:
            # Shapley-DPP and branching methods
            candidates = [{"text": b} for b in result["branches"]]
        else:
            # Unknown format - try to find any string property
            for key, value in result.items():
                if isinstance(value, str) and key not in ["problem", "ground_truth", "question"]:
                    candidates.append({"text": value})
                    break
        
        # Ensure we have at least one candidate
        if not candidates:
            candidates = [{"text": ""}]
        
        # Check each candidate for correctness
        correct_candidate_indices = []
        for j, candidate in enumerate(candidates):
            is_correct = evaluate_candidate(candidate, ground_truth, dataset)
            if is_correct:
                correct_candidate_indices.append(j)
                
        # Store example for detailed analysis
        example = {
            "index": i,
            "problem": problem,
            "ground_truth": ground_truth,
            "num_candidates": len(candidates),
            "correct_indices": correct_candidate_indices,
            "has_correct": len(correct_candidate_indices) > 0,
            "best_candidate": candidates[0] if candidates else {"text": ""}
        }
        
        if example["has_correct"]:
            metrics["correct_examples"].append(example)
        else:
            metrics["incorrect_examples"].append(example)
        
        # Calculate pass@k
        for k in k_values:
            metrics["pass_at_k"][k].append(
                1.0 if any(idx < k for idx in correct_candidate_indices) else 0.0
            )
    
    # Calculate mean pass@k
    metrics["pass_at_k_mean"] = {k: np.mean(metrics["pass_at_k"][k]) for k in k_values}
    
    # Calculate overall accuracy
    metrics["overall_accuracy"] = len(metrics["correct_examples"]) / metrics["num_problems"] if metrics["num_problems"] > 0 else 0.0
    
    # Calculate token efficiency and performance metrics if available
    # First try to use generation_time if records are available for each prompt
    if all(("generation_time" in r and "tokens_generated" in r) for r in results):
        total_time = sum(r.get("generation_time", 0) for r in results)
        total_tokens = sum(r.get("tokens_generated", 0) for r in results)
        metrics["total_time"] = total_time
        metrics["total_tokens"] = total_tokens
        metrics["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0

        # Calculate average time per prompt
        metrics["avg_time_per_prompt"] = total_time / metrics["num_problems"] if metrics["num_problems"] > 0 else 0

    # If generation_time is not available, try to use total_wallclock_time from metadata
    elif "total_wallclock_time" in metadata and "num_problems" in metadata and metadata["num_problems"] > 0:
        total_time = metadata["total_wallclock_time"]
        metrics["total_time"] = total_time

        # If we have token counts in metadata, calculate tokens per second
        if "total_tokens" in metadata and metadata["total_tokens"] > 0:
            total_tokens = metadata["total_tokens"]
            metrics["total_tokens"] = total_tokens
            metrics["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0

        # Calculate average time per prompt
        metrics["avg_time_per_prompt"] = total_time / metadata["num_problems"]

        # Get average output tokens per problem if available
        if "avg_output_tokens" in metadata:
            metrics["avg_output_tokens"] = metadata["avg_output_tokens"]
        elif "total_output_tokens" in metadata and metadata["num_problems"] > 0:
            metrics["avg_output_tokens"] = metadata["total_output_tokens"] / metadata["num_problems"]
    
    # Check if majority voting results are available
    has_majority_voting = False
    majority_voting_stats = {
        "total_problems": 0,
        "avg_confidence": 0.0,
        "problems_with_voting": 0
    }
    
    for result in results:
        if "majority_voting" in result:
            has_majority_voting = True
            majority_voting_stats["problems_with_voting"] += 1
            
            # Extract majority voting statistics
            voting_data = result["majority_voting"]
            vote_count = voting_data.get("vote_count", 0)
            total_votes = voting_data.get("total_votes", 0)
            confidence = voting_data.get("majority_confidence", 0.0)
            
            # Update stats
            if confidence > 0:
                majority_voting_stats["avg_confidence"] += confidence
    
    # Calculate average confidence
    if has_majority_voting and majority_voting_stats["problems_with_voting"] > 0:
        majority_voting_stats["avg_confidence"] /= majority_voting_stats["problems_with_voting"]
        majority_voting_stats["total_problems"] = metrics["num_problems"]
        metrics["majority_voting"] = majority_voting_stats
        
        # Add a flag to indicate majority voting was used
        metrics["has_majority_voting"] = True
    else:
        metrics["has_majority_voting"] = False

    return metrics

def evaluate_all_strategies(
    strategy_files: Dict[str, List[str]], 
    k_values: List[int], 
    dataset: str,
    specific_strategies: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all strategies or a subset of strategies.
    
    Args:
        strategy_files: Dictionary mapping strategy names to lists of file paths
        k_values: List of k values for pass@k evaluation
        dataset: Dataset name
        specific_strategies: List of specific strategies to evaluate (or None for all)
        
    Returns:
        Dictionary mapping strategy names to evaluation metrics
    """
    results = {}
    
    # Filter strategies if specific ones are requested
    if specific_strategies and specific_strategies != ["all"]:
        strategy_files = {
            strategy: files for strategy, files in strategy_files.items()
            if strategy in specific_strategies
        }
    
    # Skip generic best_of_n category if we have specific best_of_X categories
    if "best_of_n" in strategy_files:
        has_specific_best_of = any(s.startswith("best_of_") and s != "best_of_n" for s in strategy_files)
        if has_specific_best_of:
            # Skip the generic best_of_n category, we'll use the specific ones
            del strategy_files["best_of_n"]
    
    # Evaluate each strategy
    for strategy, files in strategy_files.items():
        strategy_results = []
        
        # Evaluate each file for this strategy
        for file_path in files:
            metrics = evaluate_strategy(file_path, k_values, dataset)
            strategy_results.append(metrics)
        
        # Store the strategy results
        results[strategy] = strategy_results
    
    return results

def generate_summary_table(
    all_metrics: Dict[str, List[Dict[str, Any]]],
    k_values: List[int]
) -> pd.DataFrame:
    """
    Generate a summary table of all strategy results.

    Args:
        all_metrics: Dictionary mapping strategy names to lists of evaluation metrics
        k_values: List of k values used in the evaluation

    Returns:
        DataFrame with summarized metrics
    """
    # Prepare data for the table
    rows = []
    
    # Process each strategy - aggregate results if multiple files exist for the same strategy
    for strategy, metrics_list in all_metrics.items():
        if not metrics_list:
            continue

        # For best_of_n strategies, first ensure the display name is correct using the strategy key
        if strategy.startswith("best_of_"):
            # Extract N from the strategy key (e.g., "best_of_32" → 32)
            n_value = strategy.split("_")[-1]
            
            # Update the display name to show the correct N value
            for metrics in metrics_list:
                temp = metrics["metadata"].get("temperature", 1.0)
                metrics["display_name"] = f"Best-of-{n_value} (T={temp})"

        # Use a dictionary to track unique display names
        display_name_metrics = {}

        # Group by display name
        for metrics in metrics_list:
            display_name = metrics["display_name"]
            
            # Normal handling for all strategies
            if display_name not in display_name_metrics:
                display_name_metrics[display_name] = []
            display_name_metrics[display_name].append(metrics)

        # Process each unique display name
        for display_name, metrics_group in display_name_metrics.items():
            # Calculate aggregated metrics
            total_problems = sum(m["num_problems"] for m in metrics_group)
            total_correct = sum(len(m["correct_examples"]) for m in metrics_group)

            # Create row for this display name - use the display name directly
            row = {
                "Strategy": display_name,
                "Problems": total_problems,
                "Accuracy": total_correct / total_problems if total_problems > 0 else 0
            }

            # Compute weighted average for pass@k metrics
            for k in k_values:
                weighted_pass_k = sum(
                    m["pass_at_k_mean"][k] * m["num_problems"]
                    for m in metrics_group
                ) / total_problems if total_problems > 0 else 0
                row[f"Pass@{k}"] = weighted_pass_k

            # Add average output tokens if available (independent of other performance metrics)
            if all("avg_output_tokens" in m for m in metrics_group):
                # Weighted average of output tokens
                total_output_tokens = sum(m.get("avg_output_tokens", 0) * m.get("num_problems", 0) for m in metrics_group)
                row["Avg Output Tokens"] = total_output_tokens / total_problems if total_problems > 0 else 0

            # Add other performance metrics if available - use weighted averages
            if all("tokens_per_second" in m for m in metrics_group):
                total_tokens = sum(m.get("total_tokens", 0) for m in metrics_group)
                total_time = sum(m.get("total_time", 0) for m in metrics_group)

                row["Tokens/s"] = total_tokens / total_time if total_time > 0 else 0
                row["Total Time (s)"] = total_time

                if all("avg_time_per_prompt" in m for m in metrics_group):
                    # Weighted average of time per prompt
                    row["Avg Time/Prompt (s)"] = total_time / total_problems if total_problems > 0 else 0

            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns to put performance metrics together
    preferred_order = ["Strategy", "Problems", "Accuracy"]
    # Add pass@k columns
    for k in k_values:
        preferred_order.append(f"Pass@{k}")
    # Add performance columns
    performance_metrics = ["Tokens/s", "Avg Output Tokens", "Avg Time/Prompt (s)", "Total Time (s)"]
    preferred_order.extend([col for col in performance_metrics if col in df.columns])
    # Add any remaining columns
    remaining_cols = [col for col in df.columns if col not in preferred_order]
    preferred_order.extend(remaining_cols)
    # Reorder columns that exist in the DataFrame
    df = df[[col for col in preferred_order if col in df.columns]]

    # Format the values
    for col in df.columns:
        if col in ["Strategy", "Problems"]:
            continue
        elif col in ["Tokens/s", "Total Time (s)", "Avg Time/Prompt (s)"]:
            # Format performance metrics with two decimal places
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        elif col == "Avg Output Tokens":
            # Format average output tokens with one decimal place
            df[col] = df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
        else:
            # Format accuracy metrics as percentages
            df[col] = df[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
    
    # Sort by Pass@1 (descending)
    if "Pass@1" in df.columns:
        # Convert percentage strings back to floats for sorting
        sort_values = df["Pass@1"].str.rstrip("%").astype(float) / 100
        df = df.iloc[sort_values.sort_values(ascending=False).index].reset_index(drop=True)
    
    return df

# def plot_pass_at_k_comparison(
#     all_metrics: Dict[str, List[Dict[str, Any]]], 
#     k_values: List[int], 
#     output_dir: str,
#     dataset: str,
# ):
#     """
#     Generate a pass@k comparison plot for all strategies.
    
#     Args:
#         all_metrics: Dictionary mapping strategy names to lists of evaluation metrics
#         k_values: List of k values used in the evaluation
#         output_dir: Output directory for the plot
#     """
#     plt.figure(figsize=(28, 12))
    
#     # Set up colors and markers
#     colors = plt.cm.tab10(np.linspace(0, 1, 10))
#     markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
#     # Flatten metrics for plotting
#     all_metrics_flat = []
#     for strategy, metrics_list in all_metrics.items():
#         for metrics in metrics_list:
#             all_metrics_flat.append(metrics)
    
#     # Sort by pass@1 for better visualization
#     all_metrics_flat.sort(key=lambda x: x["pass_at_k_mean"][1], reverse=True)
    
#     # Plot lines for each strategy
#     for i, metrics in enumerate(all_metrics_flat):
#         display_name = metrics["display_name"]
#         display_name = display_name + os.path.basename(metrics["file_path"]).replace(".pkl", "").replace("_raw", "")
#         # Get pass@k values
#         pass_at_k_values = [metrics["pass_at_k_mean"][k] for k in k_values]
        
#         # Plot line
#         color_idx = i % len(colors)
#         marker_idx = i % len(markers)
#         plt.plot(k_values, pass_at_k_values, marker=markers[marker_idx], 
#                  label=display_name, color=colors[color_idx])
    
#     # Customize plot
#     # plt.xscale('log')
#     plt.title(f"Pass@k Comparison Across Generation Strategies on {dataset}", fontsize=14)
#     plt.xlabel("k", fontsize=12)
#     plt.ylabel("Pass@k", fontsize=12)
#     plt.grid(alpha=0.3)
#     plt.legend(loc='lower right', fontsize=9)
    
#     # Add value labels at the right edge
#     for i, metrics in enumerate(all_metrics_flat):
#         pass_at_k_values = [metrics["pass_at_k_mean"][k] for k in k_values]
#         color_idx = i % len(colors)
#         plt.text(k_values[-1] * 1.05, pass_at_k_values[-1], 
#                 f"{pass_at_k_values[-1]:.3f}", 
#                 color=colors[color_idx],
#                 verticalalignment='center')
    
#     # Set x-axis ticks to k values
#     plt.xticks(k_values, [str(k) for k in k_values])
    
#     # Save the plot
#     os.makedirs(output_dir, exist_ok=True)
#     plot_path = os.path.join(output_dir, "pass_at_k_comparison.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Pass@k comparison plot saved to {plot_path}")

import csv

def plot_pass_at_k_comparison(
    all_metrics: Dict[str, List[Dict[str, Any]]], 
    k_values: List[int], 
    output_dir: str,
    dataset: str,
):
    """
    Generate a pass@k comparison plot for all strategies and save summary CSV.
    
    Args:
        all_metrics: Dictionary mapping strategy names to lists of evaluation metrics
        k_values: List of k values used in the evaluation
        output_dir: Output directory for the plot and CSV
        dataset: Name of dataset (for title)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plt.figure(figsize=(28, 12))
    
    # Set up colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Flatten metrics for plotting
    all_metrics_flat = []
    for strategy, metrics_list in all_metrics.items():
        for metrics in metrics_list:
            all_metrics_flat.append(metrics)
    
    # Sort by pass@1 for better visualization
    all_metrics_flat.sort(key=lambda x: x["pass_at_k_mean"][1], reverse=True)

    # Prepare CSV rows
    csv_rows = []

    # Plot lines for each strategy
    for i, metrics in enumerate(all_metrics_flat):
        display_name = metrics["display_name"]
        display_name += os.path.basename(metrics["file_path"]).replace(".pkl", "").replace("_raw", "")
        
        # Get pass@k values
        pass_at_k_values = [metrics["pass_at_k_mean"][k] for k in k_values]
        
        # Save to CSV
        row = {"label": display_name}
        for k, v in zip(k_values, pass_at_k_values):
            row[f"pass@{k}"] = v
        csv_rows.append(row)
        
        # Plot line
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        plt.plot(k_values, pass_at_k_values, marker=markers[marker_idx], 
                 label=display_name, color=colors[color_idx])
    
    # Customize plot
    plt.title(f"Pass@k Comparison Across Generation Strategies on {dataset}", fontsize=14)
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Pass@k", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)

    # Add value labels
    for i, metrics in enumerate(all_metrics_flat):
        pass_at_k_values = [metrics["pass_at_k_mean"][k] for k in k_values]
        color_idx = i % len(colors)
        plt.text(k_values[-1] * 1.05, pass_at_k_values[-1], 
                f"{pass_at_k_values[-1]:.3f}", 
                color=colors[color_idx],
                verticalalignment='center')
    
    plt.xticks(k_values, [str(k) for k in k_values])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "pass_at_k_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pass@k comparison plot saved to {plot_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "pass_at_k_summary.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["label"] + [f"pass@{k}" for k in k_values])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Pass@k summary CSV saved to {csv_path}")


def plot_performance_comparison(
    all_metrics: Dict[str, List[Dict[str, Any]]], 
    output_dir: str
):
    """
    Generate a performance comparison plot for strategies with timing information.
    
    Args:
        all_metrics: Dictionary mapping strategy names to lists of evaluation metrics
        output_dir: Output directory for the plot
    """
    # Filter metrics that have performance information
    performance_metrics = []
    for strategy, metrics_list in all_metrics.items():
        for metrics in metrics_list:
            if "tokens_per_second" in metrics:
                performance_metrics.append(metrics)
    
    if not performance_metrics:
        print("No performance metrics available for plotting")
        return
    
    # Sort by tokens per second (descending)
    performance_metrics.sort(key=lambda x: x["tokens_per_second"], reverse=True)
    
    # Prepare data for plotting
    strategy_names = [metrics["display_name"] for metrics in performance_metrics]
    tokens_per_second = [metrics["tokens_per_second"] for metrics in performance_metrics]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar chart
    bars = plt.barh(strategy_names, tokens_per_second, color=plt.cm.viridis(np.linspace(0, 0.8, len(strategy_names))))
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}", 
                ha='left', va='center')
    
    # Customize plot
    plt.title("Generation Performance Comparison", fontsize=14)
    plt.xlabel("Tokens per Second (higher is better)", fontsize=12)
    plt.ylabel("Strategy", fontsize=12)
    plt.grid(alpha=0.3, axis='x')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison plot saved to {plot_path}")

def plot_accuracy_vs_performance(
    all_metrics: Dict[str, List[Dict[str, Any]]], 
    output_dir: str
):
    """
    Generate a scatter plot of accuracy vs. performance.
    
    Args:
        all_metrics: Dictionary mapping strategy names to lists of evaluation metrics
        output_dir: Output directory for the plot
    """
    # Filter metrics that have performance information
    valid_metrics = []
    for strategy, metrics_list in all_metrics.items():
        for metrics in metrics_list:
            if "tokens_per_second" in metrics and "pass_at_k_mean" in metrics and 1 in metrics["pass_at_k_mean"]:
                valid_metrics.append(metrics)
    
    if not valid_metrics:
        print("No metrics available for accuracy vs. performance plotting")
        return
    
    # Prepare data for plotting
    strategy_names = [metrics["display_name"] for metrics in valid_metrics]
    tokens_per_second = [metrics["tokens_per_second"] for metrics in valid_metrics]
    accuracy = [metrics["pass_at_k_mean"][1] for metrics in valid_metrics]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    scatter = plt.scatter(tokens_per_second, accuracy, 
                         c=np.arange(len(valid_metrics)), 
                         cmap='viridis', 
                         s=100, 
                         alpha=0.7)
    
    # Add strategy labels
    for i, name in enumerate(strategy_names):
        plt.annotate(name, 
                    (tokens_per_second[i], accuracy[i]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Customize plot
    plt.title("Accuracy vs. Performance", fontsize=14)
    plt.xlabel("Tokens per Second", fontsize=12)
    plt.ylabel("Accuracy (Pass@1)", fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add a colorbar for visual distinction
    plt.colorbar(scatter, label="Strategy Index")
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "accuracy_vs_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy vs. Performance plot saved to {plot_path}")

def main():
    """Main function for comprehensive evaluation."""
    args = parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Handle output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle plot directory
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Find strategy result files
    print(f"Searching for strategy result files in {args.input_dir}...")
    strategy_files = find_strategy_files(args.input_dir)
    
    if not strategy_files:
        print(f"No strategy result files found in {args.input_dir}")
        return
    
    # Parse specific strategies if provided
    specific_strategies = ["all"]
    if args.strategies != "all":
        specific_strategies = [s.strip() for s in args.strategies.split(",")]
    
    # Evaluate all strategies
    print(f"Evaluating strategies: {', '.join(specific_strategies if specific_strategies != ['all'] else strategy_files.keys())}")
    evaluation_results = evaluate_all_strategies(
        strategy_files, 
        k_values, 
        args.dataset,
        specific_strategies
    )
    
    # Generate summary table
    print("Generating summary table...")
    summary_table = generate_summary_table(evaluation_results, k_values)
    print(summary_table)
    
    # Save summary table if requested
    if args.save_table:
        table_path = os.path.join(args.output_dir, "strategies_summary.csv")
        summary_table.to_csv(table_path, index=False)
        print(f"Summary table saved to {table_path}")
        
        # Also save as markdown for readability
        md_path = os.path.join(args.output_dir, "strategies_summary.md")
        with open(md_path, 'w') as f:
            f.write(f"# Generation Strategies Evaluation Summary for {args.dataset.upper()}\n\n")
            f.write(summary_table.to_markdown(index=False))
        print(f"Markdown summary saved to {md_path}")
    
    # Generate plots if requested
    if args.plot:
        print("Generating comparison plots...")
        plot_pass_at_k_comparison(evaluation_results, k_values, args.plot_dir, args.dataset)
        plot_performance_comparison(evaluation_results, args.plot_dir)
        plot_accuracy_vs_performance(evaluation_results, args.plot_dir)
    
    # Save detailed evaluation results
    detailed_path = os.path.join(args.output_dir, "strategies_evaluation_details.json")
    with open(detailed_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for strategy, metrics_list in evaluation_results.items():
            serializable_metrics = []
            for metrics in metrics_list:
                # Create a copy without numpy values
                metrics_copy = {}
                for key, value in metrics.items():
                    if key == "pass_at_k" or key == "pass_at_k_mean":
                        metrics_copy[key] = {
                            k: float(v) if isinstance(v, np.ndarray) or isinstance(v, np.number) else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, np.ndarray) or isinstance(value, np.number):
                        metrics_copy[key] = float(value)
                    elif key in ["correct_examples", "incorrect_examples"]:
                        # Just store counts to reduce file size
                        metrics_copy[key + "_count"] = len(value)
                    elif key != "metadata":  # Keep the original metadata
                        metrics_copy[key] = value
                
                serializable_metrics.append(metrics_copy)
            serializable_results[strategy] = serializable_metrics
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"Detailed evaluation results saved to {detailed_path}")
    print(f"\nEvaluation complete! Results available in {args.output_dir}")

if __name__ == "__main__":
    main()