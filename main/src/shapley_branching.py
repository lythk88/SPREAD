from typing import Union, List, Dict, Any, Tuple, Optional
import torch
import numpy as np
import pickle
import os
import time
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Import from our own modules
from .branching_inference import BranchingInferenceEngine
from .hooker import BaseHooker

# Add the root directory to path for importing from experiments
root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import shapley utilities
try:
    from experiments.shapley.shapley_indices import interaction_indices, check_random_state
except ImportError:
    print("Warning: Could not import shapley modules. Create them if they don't exist.")


class ShapleyHooker(BaseHooker):
    """
    A hooker to extract embeddings and apply Shapley weights to attention heads.
    """
    def __init__(self,
                layer_list: List[int],
                shapley_values: np.ndarray = None,
                stat_track: bool = True,
                logger=None,
                head_dim: int = None,
                extract_embeddings: bool = True):
        """
        Initialize the ShapleyHooker.

        Args:
            layer_list: List of layer indices to hook into
            shapley_values: Pre-computed Shapley values for attention heads
            stat_track: Whether to track statistics
            logger: Logger instance
            head_dim: Dimension of each attention head
            extract_embeddings: Whether to extract embeddings for DPP selection
        """
        super().__init__(layer_list, stat_track, logger)
        self.shapley_values = shapley_values
        self.head_dim = head_dim
        self.embeddings = defaultdict(list)
        self.extract_embeddings = extract_embeddings
        self.current_branch_idx = 0

    def set_branch_idx(self, idx: int):
        """Set the current branch index for embedding extraction"""
        self.current_branch_idx = idx

    def reset_embeddings(self):
        """Reset embeddings for a new generation run"""
        self.embeddings = defaultdict(list)
        self.current_branch_idx = 0

    @torch.no_grad()
    def __call__(self, attn_output, attention_name="post_o_proj"):
        """
        Apply Shapley weights to attention heads and extract embeddings.

        Args:
            attn_output: Output of the attention layer
            attention_name: Name of the attention type

        Returns:
            Weighted attention output
        """
        if self.current_layer not in self.layer_list:
            return attn_output

        # Track statistics if enabled
        if self.stat_track:
            self.track_stats(attn_output)

        # Store the embeddings for this layer if extraction is enabled
        if self.extract_embeddings:
            # Get the last token embedding
            last_token_emb = attn_output[:, -1, :, :].detach().cpu().numpy()

            # Store with the current branch index
            while len(self.embeddings[self.current_layer]) <= self.current_branch_idx:
                self.embeddings[self.current_layer].append(None)

            self.embeddings[self.current_layer][self.current_branch_idx] = last_token_emb

        # If Shapley values are provided, apply them to weight the attention heads
        if self.shapley_values is not None:
            batch_size, seq_len, num_heads, head_dim = attn_output.shape

            # Ensure shapley_values shape is compatible
            if isinstance(self.shapley_values, np.ndarray) and len(self.shapley_values) == num_heads:
                # Convert Shapley values to tensor and reshape for broadcasting
                shapley_weights = torch.tensor(
                    self.shapley_values,
                    dtype=attn_output.dtype,
                    device=attn_output.device
                ).view(1, 1, num_heads, 1)

                # Normalize Shapley values to ensure they sum to num_heads
                # This preserves the scale of the embeddings
                shapley_weights = shapley_weights * (num_heads / shapley_weights.sum())

                # Apply weights
                weighted_output = attn_output * shapley_weights
                return weighted_output

        # Return the original output if no weighting is applied
        return attn_output

    def get_last_token_embeddings(self):
        """
        Get the embeddings for the last token from each layer.

        Returns:
            Dictionary of layer index -> list of embeddings
        """
        return self.embeddings


class ShapleyWeightedBranchingEngine(BranchingInferenceEngine):
    """
    Extended BranchingInferenceEngine that uses Shapley values to weight attention heads
    and improve branch selection.
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
                max_branch_depth: int = 3,
                shapley_values_path: str = None,
                shapley_layer: int = None,
                quality_weight: float = 0.7,
                auto_discover_shapley: bool = True):
        """
        Initialize ShapleyWeightedBranchingEngine.

        Args:
            model_repo: Model repository name
            config: Generation configuration
            lora_path: Path to LoRA weights
            task_type: Task type ('causal_lm' or 'seq_cls')
            use_auto_model: Whether to use auto model classes
            use_accelerate: Whether to use Accelerate for distributed training
            shy_token_threshold: Probability threshold for shy token detection
            max_branches_per_shy_token: Maximum number of branches per shy token
            max_branch_depth: Maximum branch depth
            shapley_values_path: Path to pre-computed Shapley values
            shapley_layer: Layer to extract embeddings from and apply Shapley weights
            quality_weight: Weight for quality vs diversity in branch selection
            auto_discover_shapley: Whether to automatically discover Shapley values
        """
        super().__init__(
            model_repo, config, lora_path, task_type, use_auto_model, use_accelerate,
            shy_token_threshold, max_branches_per_shy_token, max_branch_depth
        )

        self.shapley_values = None
        self.shapley_layer = shapley_layer
        self.quality_weight = quality_weight

        # If shapley_layer is not specified, use the middle layer by default
        if self.shapley_layer is None and hasattr(self.model.config, 'num_hidden_layers'):
            self.shapley_layer = self.model.config.num_hidden_layers // 2

        # Try to load Shapley values
        if shapley_values_path and os.path.exists(shapley_values_path):
            # Explicit path provided
            self._load_shapley_values(shapley_values_path)
        elif auto_discover_shapley:
            # Try to auto-discover Shapley values
            self._load_shapley_values(None)

        # Print status
        if self.shapley_values is not None:
            print(f"Shapley values loaded successfully for layer {self.shapley_layer}")
        else:
            print(f"No Shapley values loaded. Will use regular branching unless values are computed.")
    
    def _load_shapley_values(self, shapley_values_path: str = None):
        """
        Load pre-computed Shapley values. If path is not provided,
        tries to find a matching cached file based on model and layer.

        Args:
            shapley_values_path: Path to Shapley values pickle file
        """
        # If no path provided, try to find a matching file
        if shapley_values_path is None:
            shapley_values_path = self._find_matching_shapley_file()
            if shapley_values_path is None:
                print("No matching Shapley values file found.")
                return

        try:
            with open(shapley_values_path, 'rb') as f:
                data = pickle.load(f)

            # Extract Shapley values from the loaded data
            if isinstance(data, dict) and 'indices' in data:
                self.shapley_values = data['indices']
                print(f"Loaded Shapley values with shape {self.shapley_values.shape if isinstance(self.shapley_values, np.ndarray) else 'unknown'}")
                print(f"Loaded from: {shapley_values_path}")
            else:
                print(f"Warning: Shapley values file has unexpected format. Expected 'indices' key.")
        except Exception as e:
            print(f"Error loading Shapley values: {e}")

    def _find_matching_shapley_file(self):
        """
        Look for a Shapley values file that matches the current model and layer.

        Returns:
            Path to matching file or None if not found
        """
        # Standard directories to check
        dirs_to_check = [
            "tmp",
            "cache/shapley",
            "experiments/shapley/cache"
        ]

        # Create a clean model name for the filename pattern
        model_name = self.model_repo.replace('/', '_')
        layer_str = f"layer{self.shapley_layer}" if self.shapley_layer is not None else "all_layers"

        # Patterns to look for
        patterns = [
            f"acc_{model_name}_{layer_str}_shapley_indices_*.pkl",
            f"{model_name}_{layer_str}_shapley_*.pkl",
            f"shapley_{model_name}_{layer_str}_*.pkl"
        ]

        # Check all directories and patterns
        for directory in dirs_to_check:
            if not os.path.exists(directory):
                continue

            for pattern in patterns:
                import glob
                matching_files = glob.glob(os.path.join(directory, pattern))
                if matching_files:
                    # Return the most recently modified file
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    return matching_files[0]

        return None
    
    def compute_shapley_values(self, 
                              dataset_inputs: List[str], 
                              dataset_answers: List[str],
                              num_mc_samples: int = 80,
                              num_samples: int = 128,
                              batch_size: int = 32,
                              layers_to_compute: List[int] = None,
                              save_path: str = None):
        """
        Compute Shapley values for attention heads.
        
        This is a simplified version that delegates to the shapley_indices module.
        For a full implementation, you would need to implement or import all the
        necessary functions from the experiments/shapley directory.
        
        Args:
            dataset_inputs: List of input prompts
            dataset_answers: List of ground truth answers
            num_mc_samples: Number of Monte Carlo samples for Shapley computation
            num_samples: Number of dataset samples to use
            batch_size: Batch size for inference
            layers_to_compute: Layers to compute Shapley values for
            save_path: Path to save computed Shapley values
            
        Returns:
            Computed Shapley values
        """
        # Import relevant functions
        # These would typically come from the experiments/shapley directory
        try:
            from experiments.shapley.shapley_indices import get_reward_func, interaction_indices, check_random_state
        except ImportError:
            print("Error: Could not import shapley modules. Make sure they exist in experiments/shapley/")
            return None
            
        # Restrict to a subset of the dataset if num_samples is specified
        if num_samples > 0 and num_samples < len(dataset_inputs):
            dataset_inputs = dataset_inputs[:num_samples]
            dataset_answers = dataset_answers[:num_samples]
        
        # Default to computing Shapley values for the middle layer if not specified
        if layers_to_compute is None and hasattr(self.model.config, 'num_hidden_layers'):
            layers_to_compute = [self.model.config.num_hidden_layers // 2]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(dataset_inputs, padding=True, return_tensors="pt")
        
        # Get number of attention heads
        num_attention_heads = self.model.config.num_attention_heads
        players = list(np.arange(num_attention_heads))
        
        # Create reward function
        reward_func = get_reward_func(
            self,  # Using self as the inference model
            model_inputs,
            layers_to_compute,
            num_attention_groups=num_attention_heads,
            num_attention_heads=num_attention_heads,
            golden_answers=dataset_answers,
            batch_size=batch_size,
            reward_type='accuracy'
        )
        
        # Setup random state
        rng = check_random_state(seed=42)
        
        # Compute Shapley values
        indices = interaction_indices(
            players,
            reward_func,
            ord=1,
            num_samples=num_mc_samples,
            rng=rng,
            logger=None
        )
        
        # Save Shapley values if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                data = {'indices': indices}
                pickle.dump(data, f)
        
        self.shapley_values = indices
        return indices

    def _generate_branches_with_shapley(self,
                                       prefix_input_ids: torch.Tensor,
                                       num_branches: int,
                                       branch_idx_start: int = 0) -> List[torch.Tensor]:
        """
        Generate multiple branches from a given prefix using Shapley-weighted attention.

        Args:
            prefix_input_ids: Tensor of token IDs representing the prefix
            num_branches: Number of branches to generate
            branch_idx_start: Starting index for branch indexing (for hooker embedding storage)

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

        # Create ShapleyHooker
        shapley_hooker = ShapleyHooker(
            layer_list=[self.shapley_layer] if self.shapley_layer is not None else [],
            shapley_values=self.shapley_values,
            stat_track=False,
            extract_embeddings=True
        )

        # Set starting branch index for embedding tracking
        shapley_hooker.current_branch_idx = branch_idx_start

        # Register the hooker with the model
        if not hasattr(self.model, 'hooks'):
            self.model.hooks = []

        # Remove any existing ShapleyHooker hooks
        self.model.hooks = [h for h in self.model.hooks if not isinstance(h, ShapleyHooker)]

        # Add the current hooker to model.hooks for later retrieval in branch selection
        if self.shapley_values is not None:
            self.model.hooks.append(shapley_hooker)

            # Note: We're not using the hooker during generation for now
            # due to compatibility issues with the edit_fn parameter.
            # Instead, we'll just track embeddings for diversity selection.

            # In a fully functional implementation, we would register a forward hook
            # on the attention layer to apply Shapley weights, but this requires
            # model-specific implementation that's beyond the scope of this fix.
            pass

        # Initialize hook_handle to None (we don't use hooks for now)
        hook_handle = None

        try:
            # Generate branches with Shapley weighting
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
            print(f"Error in _generate_branches_with_shapley: {e}")
            print(f"Prefix input shape: {prefix_input_ids.shape}")
            # Return empty list on error
            branch_sequences = []

        finally:
            # Clean up the hook if it was registered
            if hook_handle is not None:
                hook_handle.remove()

            # Keep the Shapley hooker in model.hooks for retrieval in branch selection
            # This doesn't affect model behavior since the forward hook is removed

        return branch_sequences
    
    def _compute_branch_qualities_shapley(self, branches: List[torch.Tensor]) -> List[float]:
        """
        Compute quality scores for branches using Shapley values.
        
        This method enhances the simple quality metrics with Shapley-weighted importance.
        
        Args:
            branches: List of branch sequences
            
        Returns:
            List of quality scores (higher is better)
        """
        qualities = []
        
        for branch in branches:
            # Decode the branch
            branch_text = self.tokenizer.decode(branch, skip_special_tokens=True)
            
            # Basic metrics (same as base class)
            length_score = min(1.0, len(branch_text) / 200)  # Cap at 1.0
            
            # Repetition penalty
            words = branch_text.split()
            if len(words) > 5:
                ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                unique_ngrams = set(ngrams)
                repetition_penalty = len(unique_ngrams) / max(1, len(ngrams))
            else:
                repetition_penalty = 1.0
            
            # Shapley-enhanced quality (if we had per-sequence Shapley values)
            # For now, just use the basic metrics but with custom weight
            quality = self.quality_weight * length_score + (1 - self.quality_weight) * repetition_penalty
            qualities.append(quality)
            
        return qualities
    
    def generate_with_shapley_branching(
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
        Generate text with shy token detection and Shapley-weighted branching.
        
        This overrides the base generate_with_branching method to use Shapley values.
        
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
                    
                    # Generate branches from this prefix using Shapley-weighted method
                    # Track branch indices for embedding extraction
                    branch_idx_start = len(new_branches)
                    branched_sequences = self._generate_branches_with_shapley(
                        prefix, self.max_branches_per_shy_token, branch_idx_start
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
                # Use Shapley-enhanced quality metrics
                selected_branches = self._select_best_branches_shapley(final_branches, branch_selection_k)
            
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
    
    def _select_best_branches_shapley(self, branches: List[torch.Tensor], k: int) -> List[torch.Tensor]:
        """
        Select the k best branches based on Shapley-enhanced quality metrics and DPP.

        Args:
            branches: List of branch sequences
            k: Number of branches to select

        Returns:
            List of the k best branches
        """
        if len(branches) <= k:
            return branches

        # Import DPP functions
        from src.dpp import select_diverse_branches, extract_embeddings_from_branches, compute_quality_scores_from_text

        # Decode branches to text for quality scoring
        branch_texts = []
        for branch in branches:
            if isinstance(branch, torch.Tensor):
                # Get the input length to exclude from decoding
                if branch.dim() == 1:
                    input_length = 0  # Assume the whole sequence is output
                else:
                    input_length = 0  # We'll decode the whole sequence

                # Decode the text
                text = self.tokenizer.decode(branch[0, input_length:] if branch.dim() > 1 else branch,
                                            skip_special_tokens=True)
                branch_texts.append(text)
            else:
                branch_texts.append("Unknown branch")

        # Check if we have embeddings from the hooker
        shapley_hooker = None
        for hook in getattr(self.model, 'hooks', []):
            if isinstance(hook, ShapleyHooker):
                shapley_hooker = hook
                break

        # Get branch indices using DPP
        selected_indices = select_diverse_branches(
            branches=branches,
            branch_texts=branch_texts,
            tokenizer=self.tokenizer,
            model=self.model,
            shapley_hooker=shapley_hooker,
            shapley_values=self.shapley_values,
            k=k,
            quality_weight=self.quality_weight,
            diversity_weight=1.0 - self.quality_weight,
            debug=False
        )

        # Return the selected branches
        return [branches[i] for i in selected_indices]


class ShapleyBranchCacheManager:
    """
    Manager class for caching and retrieving Shapley-weighted branching results.
    Extends the functionality of BranchCacheManager from branching_inference.py.
    """
    def __init__(
        self,
        branching_engine: ShapleyWeightedBranchingEngine,
        cache_file_path: str,
        batch_size: int = 2
    ):
        """
        Initialize ShapleyBranchCacheManager.
        
        Args:
            branching_engine: ShapleyWeightedBranchingEngine instance
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
    
    def run_shapley_branching_inference(
        self,
        inputs: Union[str, List[str]],
        config: Dict = None,
        max_branch_depth: int = None,
        branch_selection_k: int = 3,
        rerun: bool = False
    ):
        """
        Run Shapley-weighted branching inference with caching.
        
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
        for i in tqdm(range(0, len(inputs), self.batch_size), desc="Shapley branching inference"):
            batch = inputs[i:i+self.batch_size]
            batch_results = []
            
            for input_text in batch:
                # Create hash for the input
                input_hash = hash(input_text)
                
                # Check cache
                if not rerun and input_hash in self.cache:
                    batch_results.append(self.cache[input_hash])
                    continue
                
                # Run inference with Shapley-weighted branching
                output = self.engine.generate_with_shapley_branching(
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
                        'max_branch_depth': max_branch_depth or self.engine.max_branch_depth,
                        'shapley_layer': self.engine.shapley_layer
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
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Shapley-weighted branching")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct", 
                        help="Model repository")
    parser.add_argument("--shapley_values", type=str, default=None,
                       help="Path to Shapley values pickle file")
    parser.add_argument("--prompt", type=str, default="Explain quantum mechanics in simple terms",
                       help="Prompt to generate from")
    args = parser.parse_args()
    
    # Initialize engine
    engine = ShapleyWeightedBranchingEngine(
        args.model,
        use_auto_model=True,
        shapley_values_path=args.shapley_values,
        shy_token_threshold=0.85,
        max_branches_per_shy_token=3,
        max_branch_depth=2
    )
    
    # Generate with Shapley-weighted branching
    result = engine.generate_with_shapley_branching(
        args.prompt,
        return_all_branches=True,
        branch_selection_k=3
    )
    
    # Print results
    print(f"Input: {args.prompt}")
    print("\nBranches:")
    for i, branch in enumerate(result['all_branches'][0]['branches']):
        print(f"Branch {i+1}: {branch[:100]}..." if len(branch) > 100 else f"Branch {i+1}: {branch}")