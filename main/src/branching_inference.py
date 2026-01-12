from typing import Union, List, Dict, Any, Tuple, Optional
import torch
import numpy as np
from .inference_utils import InferenceEngine
from collections import defaultdict
import pickle
import os
import time
from tqdm import tqdm

class BranchingInferenceEngine(InferenceEngine):
    """
    Extended InferenceEngine with real-time shy token detection and branching capabilities.
    """
    def __init__(self, 
                 model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='causal_lm',
                 use_auto_model: bool = True,
                 use_accelerate: bool = False,
                 shy_token_threshold: float = 0.85,
                 max_branches_per_shy_token: int = 5,
                 max_branch_depth: int = 3):
        """
        Initialize BranchingInferenceEngine with shy token detection and branching.
        
        Args:
            model_repo: Repository name for the model
            config: Generation configuration
            lora_path: Path to LoRA adapter weights
            task_type: Task type ('causal_lm' or 'seq_cls')
            use_auto_model: Whether to use AutoModel classes
            use_accelerate: Whether to use Accelerate for distributed training
            shy_token_threshold: Probability threshold below which a token is considered "shy"
            max_branches_per_shy_token: Maximum number of branches to generate at each shy token
            max_branch_depth: Maximum depth of branching (to prevent runaway branching)
        """
        super().__init__(model_repo, config, lora_path, task_type, use_auto_model, use_accelerate)
        
        self.shy_token_threshold = shy_token_threshold
        self.max_branches_per_shy_token = max_branches_per_shy_token
        self.max_branch_depth = max_branch_depth
        self.branch_cache = {}
        
    def _is_shy_token(self, probs: torch.Tensor) -> bool:
        """
        Determine if the top token is a shy token based on its probability.
        
        Args:
            probs: Probability distribution over vocabulary
            
        Returns:
            True if the top token's probability is below the threshold
        """
        top_prob = torch.max(probs).item()
        return top_prob < self.shy_token_threshold
    
    def _generate_branches(self, prefix_input_ids: torch.Tensor, num_branches: int) -> List[torch.Tensor]:
        """
        Generate multiple branches from a given prefix.

        Args:
            prefix_input_ids: Tensor of token IDs representing the prefix
            num_branches: Number of branches to generate

        Returns:
            List of tensors containing generated branches
        """
        # Ensure prefix_input_ids has the correct shape [batch_size, seq_len]
        if prefix_input_ids.dim() == 1:
            prefix_input_ids = prefix_input_ids.unsqueeze(0)  # Add batch dimension

        # Prepare inputs from prefix
        prefix_inputs = {'input_ids': prefix_input_ids.to(self.device)}

        # Create attention mask if needed
        if prefix_input_ids.shape[0] > 0:
            attention_mask = torch.ones(
                prefix_input_ids.shape, dtype=torch.long, device=self.device
            )
            prefix_inputs['attention_mask'] = attention_mask

        # Set higher temperature for diverse branching
        branch_config = self.config.copy() if self.config else {}
        branch_config["temperature"] = branch_config.get("temperature", 1.0) * 1.2  # Increase temperature
        branch_config["num_return_sequences"] = num_branches
        branch_config["do_sample"] = True  # Ensure sampling is enabled

        # Print debug info
        print(f"Prefix shape: {prefix_input_ids.shape}")

        try:
            # Generate branches
            branches = self.model.generate(
                **prefix_inputs,
                max_new_tokens=branch_config.get("max_new_tokens", 50),
                temperature=branch_config.get("temperature", 1.2),
                top_k=branch_config.get("top_k", 50),
                top_p=branch_config.get("top_p", 0.95),
                repetition_penalty=branch_config.get("repetition_penalty", 1.0),
                num_return_sequences=num_branches,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

            # Extract the generated sequences (including prefix)
            branch_sequences = branches.sequences

        except Exception as e:
            print(f"Error in _generate_branches: {e}")
            print(f"Prefix input shape: {prefix_input_ids.shape}")
            print(f"Prefix first tokens: {prefix_input_ids[:, :5] if prefix_input_ids.dim() > 1 else prefix_input_ids[:5]}")
            # Return empty list on error
            return []

        # Optional: Could compute diversity metrics here to evaluate branch quality
        return branch_sequences
    
    def _compute_branch_qualities(self, branches: List[torch.Tensor]) -> List[float]:
        """
        Compute quality scores for branches.
        This could be extended with more sophisticated metrics.
        
        Args:
            branches: List of branch sequences
            
        Returns:
            List of quality scores (higher is better)
        """
        qualities = []
        
        for branch in branches:
            # Decode the branch
            branch_text = self.tokenizer.decode(branch, skip_special_tokens=True)
            
            # Simple metrics (as placeholders for more sophisticated ones)
            # 1. Length (longer sequences might be more complete)
            length_score = min(1.0, len(branch_text) / 200)  # Cap at 1.0
            
            # 2. Ratio of repetitive n-grams (lower is better)
            words = branch_text.split()
            if len(words) > 5:
                ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                unique_ngrams = set(ngrams)
                repetition_penalty = len(unique_ngrams) / max(1, len(ngrams))
            else:
                repetition_penalty = 1.0
                
            # Combine metrics
            quality = 0.7 * length_score + 0.3 * repetition_penalty
            qualities.append(quality)
            
        return qualities
    
    def _select_best_branches(self, branches: List[torch.Tensor], k: int) -> List[torch.Tensor]:
        """
        Select the k best branches based on quality metrics.
        
        Args:
            branches: List of branch sequences
            k: Number of branches to select
            
        Returns:
            List of the k best branches
        """
        if len(branches) <= k:
            return branches
        
        # Compute quality scores
        qualities = self._compute_branch_qualities(branches)
        
        # Get indices of top-k qualities
        top_indices = np.argsort(qualities)[-k:]
        
        # Return the corresponding branches
        return [branches[i] for i in top_indices]
    
    def generate_with_branching(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        max_branch_depth: Optional[int] = None,
        return_all_branches: bool = False,
        branch_selection_k: int = 1,
        shy_token_threshold: Optional[float] = None,
        cache_branches: bool = True,
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Generate text with shy token detection and branching.
        
        Args:
            inputs: Input text, list of texts, or tokenized inputs
            config: Generation configuration
            max_branch_depth: Override for the instance max_branch_depth
            return_all_branches: If True, return all branches instead of just the best one
            branch_selection_k: Number of branches to select (when not returning all)
            shy_token_threshold: Override for the instance shy_token_threshold
            cache_branches: Whether to cache branches for future use
            
        Returns:
            Generated text(s) or dictionary with branch information
        """
        if config is None:
            config = self.config
        
        max_branch_depth = max_branch_depth or self.max_branch_depth
        shy_token_threshold = shy_token_threshold or self.shy_token_threshold
        
        # Tokenize inputs
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, torch.Tensor):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        # Move to device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        # Initialize branch cache key based on input hash
        cache_key = str(hash(str(model_inputs['input_ids'].tolist())))
        
        # Check if we have cached results
        if cache_branches and cache_key in self.branch_cache:
            print(f"Using cached branches for key {cache_key}")
            return self.branch_cache[cache_key]
        
        # Start with autoregressive generation with probability tracking
        outputs = []
        all_branches = []
        
        for batch_idx in range(model_inputs['input_ids'].shape[0]):
            # Extract single input
            single_input = {
                'input_ids': model_inputs['input_ids'][batch_idx:batch_idx+1]
            }
            if 'attention_mask' in model_inputs:
                single_input['attention_mask'] = model_inputs['attention_mask'][batch_idx:batch_idx+1]
            
            # Initialize generation
            input_length = single_input['input_ids'].shape[1]
            current_input_ids = single_input['input_ids']
            branch_depth = 0
            branches_at_level = [[current_input_ids]]  # Start with original input
            
            # Continue branching up to max depth
            while branch_depth < max_branch_depth:
                new_branches = []
                branch_found = False
                
                # Process each branch at current level
                for branch_idx, branch in enumerate(branches_at_level[-1]):
                    # Generate tokens with probability tracking
                    generation_output = self.model.generate(
                        input_ids=branch.to(self.device),
                        max_new_tokens=config.get("max_new_tokens", 50),
                        temperature=config.get("temperature", 1.0),
                        top_k=config.get("top_k", 50),
                        top_p=config.get("top_p", 1.0),
                        repetition_penalty=config.get("repetition_penalty", 1.0),
                        num_return_sequences=1,
                        do_sample=config.get("do_sample", True),
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Extract sequences and scores
                    sequences = generation_output.sequences
                    token_scores = generation_output.scores
                    
                    # Find shy tokens
                    shy_token_positions = []

                    for pos, scores in enumerate(token_scores):
                        # Get probabilities
                        probs = torch.nn.functional.softmax(scores, dim=-1)
                        top_probs = torch.max(probs, dim=-1)[0]

                        # Detect shy tokens
                        if top_probs[0].item() < shy_token_threshold:
                            # Position in the generated sequence (excluding prefix)
                            abs_pos = input_length + pos
                            shy_token_positions.append(abs_pos)

                    if not shy_token_positions:
                        # No shy tokens found, consider this branch complete
                        new_branches.append(sequences)
                        continue

                    # For simplicity, choose the first shy token position
                    first_shy_pos = shy_token_positions[0]

                    # Check if the position is within bounds
                    if first_shy_pos >= sequences.size(1):
                        print(f"Warning: shy token position {first_shy_pos} is out of bounds for sequence of length {sequences.size(1)}")
                        new_branches.append(sequences)
                        continue

                    # Extract prefix up to shy token
                    prefix = sequences[0, :first_shy_pos+1]  # +1 to include the shy token
                    
                    # Generate branches from this prefix
                    branched_sequences = self._generate_branches(
                        prefix, self.max_branches_per_shy_token
                    )
                    
                    # Add all branches
                    for seq in branched_sequences:
                        new_branches.append(seq.unsqueeze(0))  # Add batch dimension
                    
                    branch_found = True
                
                # If no branches were found, break the loop
                if not branch_found:
                    break
                
                # Add this level's branches to all_branches
                branches_at_level.append(new_branches)
                branch_depth += 1
            
            # Collect final branches from all levels
            final_branches = []
            for level in branches_at_level:
                final_branches.extend(level)
            
            # Select best branches or return all
            if return_all_branches:
                selected_branches = final_branches
            else:
                selected_branches = self._select_best_branches(final_branches, branch_selection_k)
            
            # Decode the selected branches
            branch_texts = []
            for branch in selected_branches:
                branch_texts.append(self.tokenizer.decode(
                    branch[0, input_length:], skip_special_tokens=True
                ))
                
            if branch_selection_k == 1 and not return_all_branches:
                outputs.append(branch_texts[0] if branch_texts else "")
            else:
                outputs.append(branch_texts)
            
            # Add to all_branches
            all_branches.append({
                'input': inputs[batch_idx] if isinstance(inputs, list) else inputs,
                'branches': branch_texts,
                'branch_sequences': [b.cpu() for b in selected_branches],
            })
        
        # Prepare return value
        if return_all_branches:
            result = {
                'responses': outputs,
                'all_branches': all_branches
            }
        else:
            if branch_selection_k == 1:
                result = outputs[0] if len(outputs) == 1 else outputs
            else:
                result = outputs
        
        # Cache the result
        if cache_branches:
            self.branch_cache[cache_key] = result
            
        return result
    
    def save_branch_cache(self, filepath: str):
        """Save branch cache to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.branch_cache, f)
    
    def load_branch_cache(self, filepath: str):
        """Load branch cache from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.branch_cache = pickle.load(f)
        else:
            print(f"Warning: Branch cache file {filepath} not found.")


class BranchCacheManager:
    """
    Manager class for caching and retrieving branching results.
    """
    def __init__(
        self,
        branching_engine: BranchingInferenceEngine,
        cache_file_path: str,
        batch_size: int = 2
    ):
        """
        Initialize BranchCacheManager.
        
        Args:
            branching_engine: BranchingInferenceEngine instance
            cache_file_path: Path to the cache file
            batch_size: Batch size for processing
        """
        self.engine = branching_engine
        self.cache_file_path = cache_file_path
        self.batch_size = batch_size
        
        # Cache structure:
        # {
        #    input_hash: {
        #       'input': original_input,
        #       'branches': [branch_texts],
        #       'metadata': {additional info}
        #    }
        # }
        self.cache = {}
        
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if it exists"""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded branch cache with {len(self.cache)} entries from {self.cache_file_path}")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        with open(self.cache_file_path, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"Saved branch cache with {len(self.cache)} entries to {self.cache_file_path}")
    
    def run_branching_inference(
        self,
        inputs: Union[str, List[str]],
        config: Dict = None,
        max_branch_depth: int = None,
        branch_selection_k: int = 3,
        rerun: bool = False
    ):
        """
        Run branching inference with caching.
        
        Args:
            inputs: Input text or list of texts
            config: Generation configuration
            max_branch_depth: Maximum branch depth
            branch_selection_k: Number of branches to select
            rerun: Whether to rerun inference ignoring cache
            
        Returns:
            Dictionary with generation results
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Results to return
        results = []
        
        # Process inputs in batches
        for i in tqdm(range(0, len(inputs), self.batch_size), desc="Branching inference"):
            batch = inputs[i:i+self.batch_size]
            batch_results = []
            
            for input_text in batch:
                # Create hash for the input
                input_hash = hash(input_text)
                
                # Check cache
                if not rerun and input_hash in self.cache:
                    batch_results.append(self.cache[input_hash])
                    continue
                
                # Run inference
                output = self.engine.generate_with_branching(
                    input_text,
                    config=config,
                    max_branch_depth=max_branch_depth,
                    return_all_branches=True,
                    branch_selection_k=branch_selection_k,
                    cache_branches=False  # We're handling caching here
                )
                
                # Format result
                result = {
                    'input': input_text,
                    'branches': output['all_branches'][0]['branches'],
                    'metadata': {
                        'timestamp': time.time(),
                        'branch_selection_k': branch_selection_k,
                        'max_branch_depth': max_branch_depth or self.engine.max_branch_depth
                    }
                }
                
                # Add to cache
                self.cache[input_hash] = result
                batch_results.append(result)
            
            # Extend results
            results.extend(batch_results)
            
            # Save cache after each batch
            self._save_cache()
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize engine
    model_repo = "Qwen/Qwen2.5-Math-7B-Instruct"
    engine = BranchingInferenceEngine(
        model_repo, 
        use_auto_model=True,
        shy_token_threshold=0.85,
        max_branches_per_shy_token=3,
        max_branch_depth=2
    )
    
    # Example input
    input_text = "A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
    
    # Generate with branching
    result = engine.generate_with_branching(
        input_text,
        return_all_branches=True,
        branch_selection_k=3
    )
    
    # Print results
    print(f"Input: {input_text}")
    print("\nBranches:")
    for i, branch in enumerate(result['all_branches'][0]['branches']):
        print(f"Branch {i+1}: {branch}")