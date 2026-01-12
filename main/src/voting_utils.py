"""
Utility functions for majority voting and candidate answer processing.
These functions are shared across different generation methods like Best-of-N and SDAD.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional, Union

# Import parser functions
from src.evaluation.parser import extract_answer, strip_string


def extract_answer_from_candidate(candidate: Union[Dict[str, Any], str], dataset: str = "math") -> str:
    """
    Extract the final answer from a candidate response.
    
    Args:
        candidate: Candidate dictionary containing the response text or the text itself
        dataset: The dataset type (math, gsm8k, etc.)
        
    Returns:
        The extracted answer string
    """
    # Handle different input types
    if isinstance(candidate, dict):
        text = candidate.get("text", "")
    else:
        text = str(candidate)
    
    try:
        # First try to extract with the standard parser
        answer = extract_answer(text, data_name=dataset)
        if answer and answer.strip():
            return strip_string(answer)
            
        # If that didn't work, try more aggressive methods
        if "Answer:" in text or "answer:" in text:
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
            
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return text.strip()


def select_by_majority_vote(
    candidates: List[Union[Dict[str, Any], str]], 
    dataset: str = "math"
) -> Tuple[Any, int, int, float]:
    """
    Select the best answer from candidates using majority voting.
    
    Args:
        candidates: List of candidate responses (can be dicts with 'text' key or strings)
        dataset: The dataset type (math, gsm8k, etc.)
        
    Returns:
        tuple: (best_candidate, vote_count, total_votes, confidence)
    """
    if not candidates:
        return None, 0, 0, 0.0
        
    if len(candidates) == 1:
        return candidates[0], 1, 1, 1.0
        
    # Extract answers from all candidates
    answers = []
    for candidate in candidates:
        answer = extract_answer_from_candidate(candidate, dataset)
        answers.append(answer)
        
    # Count votes for each answer
    vote_counts = Counter(answers)
    
    # Find the most common answer
    most_common_answer, vote_count = vote_counts.most_common(1)[0]
    
    # Find the first candidate that gave this answer
    best_candidate = None
    for i, answer in enumerate(answers):
        if answer == most_common_answer:
            best_candidate = candidates[i]
            break
            
    # Calculate confidence score
    total_votes = len(candidates)
    confidence = vote_count / total_votes
    
    return best_candidate, vote_count, total_votes, confidence


def get_candidate_answers(
    candidates: List[Dict[str, Any]], 
    dataset: str = "math"
) -> Dict[str, int]:
    """
    Extract and count all unique answers from a set of candidates.
    
    Args:
        candidates: List of candidate responses
        dataset: The dataset type (math, gsm8k, etc.)
        
    Returns:
        Dictionary mapping answers to their vote counts
    """
    # Extract answers from all candidates
    answers = []
    for candidate in candidates:
        answer = extract_answer_from_candidate(candidate, dataset)
        answers.append(answer)
        
    # Count votes for each answer
    vote_counts = Counter(answers)
    
    return dict(vote_counts)


def estimate_pass_at_k(
    candidates: List[Any], 
    correct_count: Optional[int] = None,
    k: Optional[int] = None
) -> float:
    """
    Estimate the pass@k metric for a set of candidates.
    
    Args:
        candidates: List of candidate answers
        correct_count: Number of correct answers in the candidates (if None, assumes all candidates marked with is_correct)
        k: Value of k (defaults to number of candidates)
        
    Returns:
        Estimated pass@k probability
    """
    if k is None:
        k = len(candidates)
    
    k = min(k, len(candidates))
    
    # Count correct answers if not provided
    if correct_count is None:
        correct_count = sum(1 for c in candidates if (
            isinstance(c, dict) and c.get("is_correct", False)
        ))
    
    # Calculate empirical pass@k
    if correct_count == 0:
        return 0.0
    
    # Simple approximation - assumes uniform sampling without replacement
    n = len(candidates)
    c = correct_count
    
    if k >= n:
        return 1.0 if c > 0 else 0.0
    
    # Calculate binomial probability: 1 - P(no correct answers in k draws)
    # This is: 1 - (n-c choose k) / (n choose k)
    def comb(n, k):
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))
    
    if n - c < k:
        return 1.0  # Can't pick k without getting at least one correct
    
    p_no_correct = comb(n - c, k) / comb(n, k)
    return 1.0 - p_no_correct