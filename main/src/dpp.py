"""
Determinantal Point Process (DPP) implementation for diverse and high-quality subset selection.
This implementation focuses on using DPP for branch selection in the LLM generation process.
"""

import math
import time
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Union
from collections import defaultdict

# Keep the original implementation
def map_inference_dpp_greedy(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(max(0, di2s[selected_item]))
        if di_optimal == 0:
            di_optimal = epsilon
        # start_x_time = time.time()
        elements = kernel_matrix[selected_item, :]
        # print("elements: ", time.time() - start_x_time)
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        # if di2s[selected_item] < epsilon:
            # break
        selected_items.append(selected_item)

    S = np.sort(np.array(selected_items))
    return S, np.linalg.det(kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)])

def map_inference_dpp_local_search_2(L, k, verbose=False):
    start_time = time.time()
    greedy_sol, greedy_prob = map_inference_dpp_greedy(L, k)
    greedy_time = time.time() - start_time

    if verbose:
        print("Prob: ", greedy_prob)

    cur_sol = greedy_sol.copy()
    cur_prob = greedy_prob
    obj_greedy = greedy_prob

    N = L.shape[0]
    all_idx = np.array(range(N))
    ns_idx = np.setdiff1d(all_idx, cur_sol)

    # L = np.arange(100).reshape(10, 10)
    L_S = L[cur_sol[:, np.newaxis], cur_sol]
    it = 0

    while True:
        start_iter_time = time.time()

        idx = np.array(range(len(cur_sol)))
        best_removal_idx = 0
        best_removal_prob = 0

        for i in range(len(cur_sol)):
            # cur_sol[i], cur_sol[-1] = cur_sol[-1], cur_sol[i]
            idx[i], idx[-1] = idx[-1], idx[i]
            L_Se = L_S[idx[:-1, np.newaxis], idx[:-1]]
            prob = np.linalg.det(L_Se)

            if prob > best_removal_prob:
                best_removal_idx = i
                best_removal_prob = prob
        obj_loc = best_removal_prob

        brid = best_removal_idx
        br = cur_sol[brid]

        best_neighbors = cur_sol.copy()
        best_add = -1
        best_neighbors_prob = cur_prob
        localopt = True

        for v in ns_idx:
            cur_sol[brid] = v
            L_S[brid, :] = L[v, cur_sol]
            L_S[:, brid] = L[cur_sol, v]
            prob = np.linalg.det(L_S)

            if prob > best_neighbors_prob:
                best_neighbors_prob = prob
                best_add = v
                localopt = False

        if verbose:
            print("Iter {}:".format(it))
            print("remove item: ", br)
            print("add item: ", best_add)
            print("best_neighbors_prob: ", best_neighbors_prob)

        if not localopt:
            cur_sol[brid] = best_add
            cur_prob = best_neighbors_prob
            L_S[brid, :] = L[best_add, cur_sol]
            L_S[:, brid] = L[cur_sol, best_add]
            ns_idx = np.setdiff1d(all_idx, cur_sol)
        else:
            cur_sol = best_neighbors
            cur_prob = best_neighbors_prob
            break
        it += 1

    ls_time = time.time() - start_time
    return cur_sol, obj_loc, ls_time, greedy_sol, greedy_prob, greedy_time


def select_k_samples_with_dpp_from_dataset(embeddings, subsample_size):
    assert 0 < subsample_size < embeddings.shape[0], f"subsample_size must be less than the number of embeddings ({embeddings.shape[0]})"

    print(f"Applying k-medoids clustering to sample {subsample_size} points out of {embeddings.shape[0]} dimension ({embeddings.shape[1]})...")
    cur_sol, obj_loc, ls_time, greedy_sol, greedy_prob, greedy_time = map_inference_dpp_local_search_2(embeddings @ embeddings.T, subsample_size, verbose=False)
    return cur_sol


# Enhanced DPP implementations for branching

def compute_kernel_matrix(
    features: np.ndarray,
    quality_scores: np.ndarray = None,
    similarity_bandwidth: float = 1.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute the DPP kernel matrix from feature vectors and quality scores.

    The kernel combines quality (diagonal) and diversity (off-diagonal) information.
    K = diag(q) * S * diag(q), where S is similarity and q is quality.

    Args:
        features: Feature vectors of shape (n_items, n_features)
        quality_scores: Quality scores of shape (n_items,), if None, all set to 1.0
        similarity_bandwidth: Bandwidth parameter for RBF similarity
        normalize: Whether to normalize the kernel matrix

    Returns:
        Kernel matrix of shape (n_items, n_items)
    """
    n_items = features.shape[0]

    # Default quality scores to 1.0 if not provided
    if quality_scores is None:
        quality_scores = np.ones(n_items)

    # Ensure quality scores are positive
    quality_scores = np.maximum(quality_scores, 0.0)

    # Compute similarity matrix using dot product (fast)
    similarity = features @ features.T

    # Apply RBF kernel if desired (slower but may capture similarity better)
    if similarity_bandwidth is not None:
        # Compute squared distances
        norms = np.sum(features**2, axis=1, keepdims=True)
        sq_dists = norms + norms.T - 2 * similarity

        # Apply RBF kernel
        similarity = np.exp(-sq_dists / (2 * similarity_bandwidth**2))

    # Construct quality-weighted kernel
    # K = diag(q) * S * diag(q)
    quality_diag = np.diag(quality_scores)
    kernel = quality_diag @ similarity @ quality_diag

    # Normalize if requested
    if normalize and np.max(kernel) > 0:
        kernel = kernel / np.max(kernel)

    return kernel


def dpp_select_diverse_subset(
    features: np.ndarray,
    quality_scores: np.ndarray = None,
    k: int = None,
    similarity_bandwidth: float = 1.0,
    method: str = 'greedy'
) -> List[int]:
    """
    Select a diverse and high-quality subset using DPP.

    Args:
        features: Feature vectors of shape (n_items, n_features)
        quality_scores: Quality scores of shape (n_items,)
        k: Number of items to select
        similarity_bandwidth: Bandwidth parameter for RBF similarity
        method: Selection method ('greedy' or 'local_search')

    Returns:
        List of selected item indices
    """
    if features.shape[0] <= k:
        # If we have fewer items than k, return all indices
        return list(range(features.shape[0]))

    # Compute kernel matrix
    kernel = compute_kernel_matrix(
        features=features,
        quality_scores=quality_scores,
        similarity_bandwidth=similarity_bandwidth
    )

    # Apply selected method
    if method == 'greedy':
        selected, _ = map_inference_dpp_greedy(kernel, k)
        return selected.tolist()
    elif method == 'local_search':
        selected, _, _, _, _, _ = map_inference_dpp_local_search_2(kernel, k)
        return selected.tolist()
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_embeddings_from_branches(
    branches: List[torch.Tensor],
    tokenizer,
    model,
    shapley_hooker=None,
    layer_idx: int = None,
    use_shapley_weights: bool = True
) -> np.ndarray:
    """
    Extract embeddings from branches for DPP selection.

    Args:
        branches: List of branch sequences (token IDs)
        tokenizer: Tokenizer for decoding
        model: Model for computing embeddings
        shapley_hooker: Optional ShapleyHooker to use for embedding extraction
        layer_idx: Layer index to extract embeddings from
        use_shapley_weights: Whether to apply Shapley weights

    Returns:
        Array of embeddings for each branch
    """
    embeddings = []

    # If no ShapleyHooker is provided, use mean pooling of final layer outputs
    if shapley_hooker is None or not hasattr(shapley_hooker, 'embeddings') or not shapley_hooker.embeddings:
        for branch in branches:
            # Convert to tensor if necessary
            if not isinstance(branch, torch.Tensor):
                branch = torch.tensor(branch, device=model.device)

            # Ensure the branch has a batch dimension
            if branch.dim() == 1:
                branch = branch.unsqueeze(0)

            # Get model outputs
            with torch.no_grad():
                outputs = model(branch)

            # Get the final layer hidden states
            hidden_states = outputs.last_hidden_state

            # Mean pooling
            embedding = hidden_states.mean(dim=1).detach().cpu().numpy()
            embeddings.append(embedding.flatten())

    # If ShapleyHooker is provided, use the embeddings it extracted
    else:
        # Get embeddings from the hooker
        hooker_embeddings = shapley_hooker.embeddings

        # If a specific layer is requested, use that
        if layer_idx is not None and layer_idx in hooker_embeddings:
            for i, branch in enumerate(branches):
                if i < len(hooker_embeddings[layer_idx]):
                    embedding = hooker_embeddings[layer_idx][i].flatten()
                    embeddings.append(embedding)
                else:
                    # Fallback for missing embeddings
                    embeddings.append(np.zeros(hooker_embeddings[layer_idx][0].shape).flatten())

        # Otherwise, use the first available layer
        elif hooker_embeddings:
            layer_idx = next(iter(hooker_embeddings.keys()))
            for i, branch in enumerate(branches):
                if i < len(hooker_embeddings[layer_idx]):
                    embedding = hooker_embeddings[layer_idx][i].flatten()
                    embeddings.append(embedding)
                else:
                    # Fallback for missing embeddings
                    embeddings.append(np.zeros(hooker_embeddings[layer_idx][0].shape).flatten())

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # If no embeddings were extracted, create dummy embeddings
    if len(embeddings) == 0 or embeddings.size == 0:
        embeddings = np.random.randn(len(branches), 10)

    return embeddings


def compute_quality_scores_from_text(
    branch_texts: List[str],
    shapley_values: np.ndarray = None,
    quality_metrics: Dict[str, float] = None
) -> np.ndarray:
    """
    Compute quality scores for branches based on text features.

    Args:
        branch_texts: List of branch text strings
        shapley_values: Shapley values to use for weighting (optional)
        quality_metrics: Dictionary mapping metric names to weights

    Returns:
        Array of quality scores
    """
    n_branches = len(branch_texts)
    scores = np.ones(n_branches)

    # Define default quality metrics if none provided
    if quality_metrics is None:
        quality_metrics = {
            'length': 0.5,         # Longer answers might be more complete
            'coherence': 0.3,      # Approximated by sentence count
            'repetition': -0.2     # Penalize repetition
        }

    # Calculate basic quality metrics
    for i, text in enumerate(branch_texts):
        # Length score (normalized)
        length_score = min(1.0, len(text) / 1000)

        # Coherence score (approximated by sentence count)
        sentences = [s for s in text.split('.') if s.strip()]
        coherence_score = min(1.0, len(sentences) / 10)

        # Repetition score (penalty for repeated words)
        words = text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_score = unique_words / len(words)
        else:
            repetition_score = 0.0

        # Combined score
        scores[i] = (
            quality_metrics.get('length', 0.5) * length_score +
            quality_metrics.get('coherence', 0.3) * coherence_score +
            quality_metrics.get('repetition', -0.2) * (1 - repetition_score)
        )

    # Normalize scores
    if np.max(scores) > 0:
        scores = scores / np.max(scores)

    return scores


def select_diverse_branches(
    branches: List[torch.Tensor],
    branch_texts: List[str],
    tokenizer,
    model,
    shapley_hooker=None,
    shapley_values: np.ndarray = None,
    k: int = 5,
    quality_weight: float = 0.7,
    diversity_weight: float = 0.3,
    debug: bool = False
) -> List[int]:
    """
    Select diverse and high-quality branches using DPP.

    Args:
        branches: List of branch sequences (token IDs)
        branch_texts: List of branch text strings
        tokenizer: Tokenizer for decoding
        model: Model for computing embeddings
        shapley_hooker: Optional ShapleyHooker for embedding extraction
        shapley_values: Shapley values to use for weighting
        k: Number of branches to select
        quality_weight: Weight for quality in selection
        diversity_weight: Weight for diversity in selection
        debug: Whether to print debug information

    Returns:
        List of selected branch indices
    """
    if len(branches) <= k:
        # If we have fewer branches than k, return all indices
        return list(range(len(branches)))

    # Extract embeddings for diversity measurement
    embeddings = extract_embeddings_from_branches(
        branches=branches,
        tokenizer=tokenizer,
        model=model,
        shapley_hooker=shapley_hooker
    )

    # Compute quality scores
    quality_scores = compute_quality_scores_from_text(
        branch_texts=branch_texts,
        shapley_values=shapley_values
    )

    # Combine quality and diversity using weights
    weighted_scores = quality_weight * quality_scores

    if debug:
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Quality scores: {quality_scores}")
        print(f"Weighted scores: {weighted_scores}")

    # Select diverse subset
    selected_indices = dpp_select_diverse_subset(
        features=embeddings,
        quality_scores=weighted_scores,
        k=k,
        method='greedy'
    )

    return selected_indices

def shapley_dpp_sampling(
    model,
    prefix_text: str,
    tokenizer,
    token_probs: torch.Tensor,
    top_k_tokens: int = 50,
    num_candidates: int = 5,
    layer_name: str = None,
    quality_weight: float = 0.7,
    similarity_bandwidth: float = 0.5,
    debug: bool = False
) -> torch.Tensor:
    """
    Sample diverse tokens from the token probability distribution using Shapley-weighted DPP.

    This function extracts token embeddings from the model and uses DPP to sample
    a diverse and high-quality subset of tokens.

    Args:
        model: The language model
        prefix_text: The text prefix to condition on
        tokenizer: Tokenizer for the model
        token_probs: Token probability distribution at the sampling position
        top_k_tokens: Number of top tokens to consider
        num_candidates: Number of tokens to sample
        layer_name: Layer to extract embeddings from
        quality_weight: Weight for quality in DPP selection
        similarity_bandwidth: Bandwidth parameter for similarity kernel
        debug: Whether to print debug information

    Returns:
        Tensor of selected token indices
    """
    # Get device
    device = next(model.parameters()).device

    # Sort token probabilities to get top-k
    top_k_probs, top_k_indices = torch.topk(token_probs, min(top_k_tokens, token_probs.shape[-1]))

    # Convert probabilities to quality scores
    quality_scores = top_k_probs.cpu().numpy()

    # Create input for each candidate token
    candidate_inputs = []
    for token_idx in top_k_indices:
        token_id = token_idx.item()
        # Create input with one token added to the prefix
        prefix_ids = tokenizer.encode(prefix_text, return_tensors="pt").to(device)
        # Append the candidate token
        input_ids = torch.cat([prefix_ids, torch.tensor([[token_id]]).to(device)], dim=1)
        candidate_inputs.append(input_ids)
    
    # Get embeddings for each candidate
    token_embeddings = []
    with torch.no_grad():
        for input_ids in candidate_inputs:
            # Forward pass through the model
            outputs = model(input_ids)
            
            # Get hidden states from the specified layer if available
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None and layer_name is not None:
                try:
                    # Find the layer index based on the name
                    layer_idx = int(layer_name.split('.')[-1]) if isinstance(layer_name, str) and '.' in layer_name else 0
                    if layer_idx < len(outputs.hidden_states):
                        hidden_state = outputs.hidden_states[layer_idx]
                    else:
                        hidden_state = outputs.last_hidden_state
                except:
                    # Fallback to last hidden state
                    hidden_state = outputs.last_hidden_state
            else:
                # Fallback to last hidden state
                hidden_state = outputs.last_hidden_state
                
            # Get the last token embedding (the candidate token)
            embedding = hidden_state[:, -1, :].cpu().numpy()
            token_embeddings.append(embedding.flatten())
    
    # Convert to numpy array
    token_embeddings = np.array(token_embeddings)
    
    if debug:
        print(f"Token embeddings shape: {token_embeddings.shape}")
        print(f"Quality scores shape: {quality_scores.shape}")
    
    # Compute kernel matrix
    kernel = compute_kernel_matrix(
        features=token_embeddings,
        quality_scores=quality_scores,
        similarity_bandwidth=similarity_bandwidth,
        normalize=True
    )
    
    # Select diverse subset
    if token_embeddings.shape[0] <= num_candidates:
        selected_indices = list(range(token_embeddings.shape[0]))
    else:
        selected, _ = map_inference_dpp_greedy(kernel, num_candidates)
        selected_indices = selected.tolist()
    
    # Map back to the original token indices
    sampled_tokens = top_k_indices[selected_indices]
    
    return sampled_tokens