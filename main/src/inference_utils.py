from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, List, Dict, Any, Callable
import json
from pathlib import Path
import torch 
import transformers 
import numpy as np
from peft import PeftModel
import os 
from src.hooker import BaseHooker
import torch.nn.functional as F

import pickle


class CacheManager:
    def __init__(
        self,
        inference_engine,        # an instance of InferenceEngine
        cache_file_path: str,
        batch_size: int = 2
    ):
        """
        CacheManager class for managing cached responses from an InferenceEngine using pickle.
        
        Args:
            inference_engine: an instance of the InferenceEngine class.
            cache_file_path: str, path to the pickle file where cached results will be stored.
            batch_size: int, number of inputs processed per batch.
        """
        self.engine = inference_engine
        self.cache_file_path = cache_file_path
        self.batch_size = batch_size
        
        # Internal in-memory store for caching
        # Each list index corresponds to an input index:
        #   - responses: List[str]
        #   - topk_probs: List[np.ndarray] (each is shape (seq_len, topk))
        #   - topk_tokens: List[List[List[str]]] (per input: shape (seq_len, topk))
        self.cache_data = {
            "responses": [],
            "topk_probs": [],
            "topk_tokens": []
        }
        
        self._load_cache()

    def _load_cache(self):
        """
        Load cache from disk if it exists (pickle format).
        """
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                self.cache_data = loaded_data
            except Exception as e:
                print(f"[CacheManager] Error loading cache file. Overwriting with fresh cache. Error: {e}")
                self.cache_data = {
                    "responses": [],
                    "topk_probs": [],
                    "topk_tokens": []
                }

    def _save_cache(self):
        """
        Save the current cache to disk in pickle format.
        """
        with open(self.cache_file_path, 'wb') as f:
            pickle.dump(self.cache_data, f)

    def run_inference(
        self,
        inputs,          # List[str] or single str
        topk: int = 5,
        rerun: bool = False
    ):
        """
        Runs inference on the provided inputs using the InferenceEngine, utilizing caching via pickle.

        Args:
            inputs: str or List[str]. If it's a single string, we treat it as len=1.
            topk: number of top-k tokens to retrieve per step.
            rerun: bool, if True, ignores the cache and re-runs everything from scratch.

        Returns:
            A tuple (all_responses, all_topk_probs, all_topk_tokens) with the same length as inputs.
        """
        
        # Normalize input to a list of strings
        if isinstance(inputs, str):
            inputs = [inputs]
        
        total_inputs = len(inputs)

        # If rerun is True, clear the in-memory cache and proceed
        if rerun:
            self.cache_data = {
                "responses": [],
                "topk_probs": [],
                "topk_tokens": []
            }
        
        cached_len = len(self.cache_data["responses"])
        
        # If we already have enough cached results and no rerun, return them directly
        if not rerun and cached_len >= total_inputs:
            all_responses = self.cache_data["responses"][:total_inputs]
            all_topk_probs = self.cache_data["topk_probs"][:total_inputs]
            all_topk_tokens = self.cache_data["topk_tokens"][:total_inputs]
            return all_responses, all_topk_probs, all_topk_tokens
        
        # Infer only the missing portion (from cached_len to total_inputs)
        start_idx = 0 if rerun else cached_len
        while start_idx < total_inputs:
            print(f'gen {start_idx} / {total_inputs}')
            end_idx = min(start_idx + self.batch_size, total_inputs)
            batch_inputs = inputs[start_idx:end_idx]

            # Generate using your InferenceEngine
            (
                batch_responses,
                batch_topk_values,
                batch_topk_indices,
                batch_topk_tokens
            ) = self.engine.generate_with_topk_probs(
                batch_inputs,
                config=self.engine.config,
                topk=topk
            )

            # Also record token counts if available
            if hasattr(self.engine, 'last_token_counts'):
                print(f"Recording token counts: {self.engine.last_token_counts}")

                if not hasattr(self, 'token_counts'):
                    self.token_counts = {
                        'input_tokens': [],
                        'output_tokens': [],
                        'total_tokens': []
                    }

                batch_size = len(batch_responses)
                for i in range(batch_size):
                    # For each item in the batch, store the token counts
                    if isinstance(self.engine.last_token_counts['input_tokens'], int):
                        # Same count for all items in batch
                        self.token_counts['input_tokens'].append(self.engine.last_token_counts['input_tokens'])
                        self.token_counts['output_tokens'].append(self.engine.last_token_counts['output_tokens'])
                        self.token_counts['total_tokens'].append(self.engine.last_token_counts['total_tokens'])
                    else:
                        # Different counts per item
                        self.token_counts['input_tokens'].append(self.engine.last_token_counts['input_tokens'][i])
                        self.token_counts['output_tokens'].append(self.engine.last_token_counts['output_tokens'][i])
                        self.token_counts['total_tokens'].append(self.engine.last_token_counts['total_tokens'][i])

            # Update our cache
            for i in range(len(batch_responses)):
                self.cache_data["responses"].append(batch_responses[i])
                self.cache_data["topk_probs"].append(batch_topk_values[i])
                self.cache_data["topk_tokens"].append(batch_topk_tokens[i])

            # Save cache after each batch
            self._save_cache()
            
            start_idx = end_idx

        # Retrieve full results up to total_inputs
        all_responses = self.cache_data["responses"][:total_inputs]
        all_topk_probs = self.cache_data["topk_probs"][:total_inputs]
        all_topk_tokens = self.cache_data["topk_tokens"][:total_inputs]

        return all_responses, all_topk_probs, all_topk_tokens
        

class InferenceEngine:
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='causal_lm',
                 use_auto_model: bool = True,
                 use_accelerate: bool = False):
        
        self.model_repo = model_repo
        self.use_accelerate = use_accelerate
        
        self.model_name = model_repo.split('/')[-1]
        self.default_generation_config_path = "config/default_generation_config.json"
        
        if config is not None:
            self.config = config
        else:
            self.config = self.load_config(self.default_generation_config_path)
        
            
        self.init_model_tokenizer(model_repo, 
                                lora_path=lora_path,
                                use_auto_model=use_auto_model,
                                task_type=task_type,
                                attention_implementation=self.config.get("attention_implementation", 'eager'),
                                use_accelerate=use_accelerate)

            
    def init_model_tokenizer(self,
                             model_repo: str,
                             lora_path: str = None, 
                             use_auto_model: bool = True,
                             task_type: str = 'causal_lm',
                             attention_implementation: str = 'eager',
                             use_accelerate: bool = False):
        
        if task_type not in {'seq_cls', 'causal_lm'}:
            raise ValueError(f"Task type {task_type} not supported.")
        if attention_implementation not in {'eager', 'flash_attention_2'}:
            raise ValueError(f"Attention implementation {attention_implementation} not supported.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        if use_auto_model:
            if task_type == 'causal_lm':
                self.model = AutoModelForCausalLM.from_pretrained(model_repo if lora_path is None else lora_path, 
                                                            torch_dtype=torch.float16, 
                                                            device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                            attn_implementation=attention_implementation)
                
            elif task_type == 'seq_cls':
                self.model = AutoModelForSequenceClassification.from_pretrained(model_repo if lora_path is None else lora_path, 
                                                            torch_dtype=torch.float16, 
                                                            device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                            attn_implementation=attention_implementation)
        else:
            if 'qwen' in model_repo.lower():
                from src.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
                self.model = Qwen2ForCausalLM.from_pretrained(model_repo, 
                                                        torch_dtype=torch.float16, 
                                                        device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                        attn_implementation=attention_implementation)
                if lora_path is not None:
                    # self.add_special_tokens()
                    self.model = PeftModel.from_pretrained(
                                self.model,
                                lora_path,
                                is_trainable=False,
                            )
                
                
            elif 'gemma' in model_repo.lower():
                raise NotImplementedError()
                # from .models.gemma2.modeling_gemma2 import GemmaForCausalLM
                # self.model = GemmaForCausalLM.from_pretrained(model_repo, 
                #                                         torch_dtype=torch.float16, 
                #                                         device_map='auto' if not use_accelerate else self.distributed_state.device,
                #                                         attn_implementation=attention_implementation)
            else:
                raise ValueError(f"Model {model_repo} not supported. or set use_auto_model=True instead")
        
        
        print(f'Model loaded from {model_repo if lora_path is None else lora_path}')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model.eval()
        self.device = self.model.device if not use_accelerate else self.distributed_state.device
    
    @staticmethod
    def save_config(config: Dict, filepath: str = "config/default_generation_config.json"):
        """Save generation configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod    
    def load_config(filepath: str = "config/default_generation_config.json") -> Dict:
        """Load generation configuration from JSON file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Config file not found at {filepath}.")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        return_raw_output: bool = False,
    ):
        if config is None:
            config = self.config
        print(f"Using generation config: {config}")
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")

        # Track input token counts (prompt tokens)
        input_token_count = model_inputs['input_ids'].shape[1]
        seq_len = input_token_count
        model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}

        # Set return_dict_in_generate to True to get sequences in output object
        return_dict_in_generate = True
        output = self.model.generate(
            **model_inputs,
            max_new_tokens=config.get("max_new_tokens", 2048),
            temperature=config.get("temperature", 1.0),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 1.0),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            num_return_sequences=config.get("num_return_sequences", 1),
            do_sample=config.get("do_sample", True),
            use_cache=config.get("use_cache", True),
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=True,  # Enable scores to get probabilities
        )
        
        token_ids_list = output.sequences[0, model_inputs["input_ids"].shape[1]:]
        token_probs_list = [F.softmax(score[0], dim=-1)[token_id].item() for score, token_id in zip(output.scores, token_ids_list)]


        # Track output token counts (generated tokens)
        if return_dict_in_generate:
            output_sequences = output.sequences
            output_token_count = output_sequences.shape[1] - seq_len
        else:
            output_token_count = output.shape[1] - seq_len

        # Store token counts for access by other methods
        self.last_token_counts = {
            'input_tokens': input_token_count,
            'output_tokens': output_token_count,
            'total_tokens': input_token_count + output_token_count
        }

        if return_raw_output:
            return output
        else:
            if return_dict_in_generate:
                response = self.tokenizer.batch_decode(output.sequences[:, seq_len:], skip_special_tokens=True)
            else:
                response = self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True)

            # Return response with token counts for convenience
            result = {
                "text": response,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "token_count": input_token_count + output_token_count
            }
            return result, token_probs_list, token_ids_list
    
    def generate_with_topk_probs(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        topk: int = 5,
    ):

        if config is None:
            config = self.config

        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")

        # Track input token counts (prompt tokens)
        input_token_count = model_inputs['input_ids'].shape[1]
        seq_len = input_token_count
        model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}

        output = self.model.generate(
            **model_inputs,
            max_new_tokens=config.get("max_new_tokens", 50),
            temperature=config.get("temperature", 1.0),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 1.0),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            num_return_sequences=config.get("num_return_sequences", 1),
            do_sample=config.get("do_sample", True),
            use_cache=config.get("use_cache", True),
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

        # Track output token counts (generated tokens)
        output_sequences = output.sequences
        output_token_count = output_sequences.shape[1] - seq_len

        # Store token counts for access by other methods
        self.last_token_counts = {
            'input_tokens': input_token_count,
            'output_tokens': output_token_count,
            'total_tokens': input_token_count + output_token_count
        }

        probs = []
        topk_values = []
        topk_indices = []

        # Iterate over each timestep's score matrix
        for timestep_scores in output.scores:
            # Apply softmax to calculate probabilities for the current timestep
            timestep_probs = torch.nn.functional.softmax(timestep_scores, dim=-1)
            # Get the top-k probabilities and indices for the current timestep
            timestep_topk_values, timestep_topk_indices = torch.topk(timestep_probs, k=topk, dim=-1)
            # Append to lists
            probs.append(timestep_probs.cpu().numpy())
            topk_values.append(timestep_topk_values.cpu().numpy())
            topk_indices.append(timestep_topk_indices.cpu().numpy())

        # Concatenate the top-k values and indices along the sequence length dimension
        # This creates arrays of shape (batch_size, seq_length, topk)
        topk_values = np.stack(topk_values, axis=1)  # Shape: (batch_size, seq_length, topk)
        topk_indices = np.stack(topk_indices, axis=1)  # Shape: (batch_size, seq_length, topk)

        # Decode the generated sequences
        response1 = self.tokenizer.batch_decode(output.sequences[:, :], skip_special_tokens=True)

        response = self.tokenizer.batch_decode(output.sequences[:, seq_len:], skip_special_tokens=True)

        # Decode the top-k tokens for each timestep
        topk_tokens = []
        for batch_idx in range(topk_indices.shape[0]):  # Iterate over batch
            batch_tokens = []
            for timestep_idx in range(topk_indices.shape[1]):  # Iterate over timesteps
                # Decode tokens for the current timestep
                timestep_tokens = [
                    self.tokenizer.decode(topk_indices[batch_idx, timestep_idx, token_idx])
                    for token_idx in range(topk_indices.shape[2])
                ]
                batch_tokens.append(timestep_tokens)
            topk_tokens.append(batch_tokens)
            
        return response, topk_values, topk_indices, topk_tokens
        
    def forward(
        self,
        inputs: Union[str, List[str], Dict],
    ) -> str:
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        
        model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}
        output = self.model(**model_inputs, use_cache=False)
        
        return output

    @torch.no_grad()
    def get_attention_at_layers(
        self,
        inputs: Union[str, List[str], Dict],
        layers_to_prune: List[int] = [3],
        stat_track: bool = True,
    ) -> str:
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        attention_hooker = BaseHooker(layer_list=layers_to_prune, stat_track=stat_track)
        
        output = self.model(**model_inputs, 
                            edit_fn=attention_hooker,
                            use_cache=False)
        attention_hooker.log_stats()  
        print(f"Attention hooker has been called {attention_hooker.__call__.call_count} time(s).")
        return attention_hooker.attention



def test_inference_engine():
    model_repo = "Qwen/Qwen2.5-Math-7B-Instruct"
    model_name = model_repo.split('/')[-1].lower()
    engine = InferenceEngine(model_repo, use_auto_model=True)
    from src.prompts.qwen import MATH_PROMPT_TEMPLATE
    question1 = r"Define \[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write \[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$"
    question2 = r"A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
    prompt1 = MATH_PROMPT_TEMPLATE.format(input=question1)
    prompt2 = MATH_PROMPT_TEMPLATE.format(input=question2)
    
    
    dataset_name = 'dummy'
    cache_path = f"cache/{dataset_name}_{model_name}.pkl"
    
    cache_manager = CacheManager(inference_engine=engine, cache_file_path=cache_path, batch_size=4)
    responses, topk_probs, topk_tokens = cache_manager.run_inference([prompt1, prompt2], topk=5, rerun=False)
    print(responses, topk_probs)


if __name__ == '__main__':
    # log_folder = '../log'
    # os.makedirs(log_folder, exist_ok=True)
    # log_file = os.path.join(log_folder, 'transformers_utils.log')
    # logging.basicConfig(
    #     filename=log_file,
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s"
    # )
    test_inference_engine()
    # os.makedirs("tmp", exist_ok=True)
    # model_repo = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # inference_transformers = InferenceEngine(model_repo, use_auto_model=True)
    # res = inference_transformers.generate('I love you baby')
    # print(res)
    # breakpoint()
    # data_path = f"notebook/tmp/{model_repo.replace('/', '_')}_generated_outputs_1batch.pkl"
    # # data_path = data_path.replace("1.5B", "7B")
    # with open(data_path, "rb") as in_f:
    #     attention_data_base = pickle.load(in_f)
    
    # attention_data_base['input_ids'] = attention_data_base['input_ids']
    # attention_data_base['labels'] = attention_data_base['labels']
    # default_config = InferenceTransformers.load_config('config/default_generation_config.json')
    
    # model_inputs = inference_transformers.generate(attention_data_base, 
    #                                                return_raw_output=False,
    #                                                heads_to_prune=[],
    #                                                layers_to_prune=[],)
    # print(model_inputs)
    # from grading import grader 
    
    # output = inference_transformers.forward(inputs=attention_data_base,
    #                                         heads_to_prune=[3],
    #                                         layers_to_prune=[3], 
    #                                         stat_track=True)
    # output = inference_transformers.get_attention_at_layers(inputs=attention_data_base, layers_to_prune=[3])
    
    # print(output)
    # text = "Square root of 64 is "
    # output = inference_transformers.generate(text)
    # print(output)
    # breakpoint()
    # logging.info('Testing single text ')
    # logging.info(inference_transformers.generate(text, return_raw_output=False))
    # logging.info(inference_transformers.generate(text, return_raw_output=True).shape)
    # logging.info('Testing multiple texts ')
    # logging.info(inference_transformers.generate(["Hello, I am a", "you are dead to me!"], return_raw_output=False))
    # logging.info(inference_transformers.generate(["Hello, I am a", "you are dead to me!"], return_raw_output=True).shape)
    # logging.info('Testing tokenized text ')
    # model_inputs = inference_transformers.tokenizer(text, return_tensors="pt")
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=False))
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=True).shape)
    # logging.info('Testing batch tokenized text ')
    # model_inputs = inference_transformers.tokenizer(["Hello, I am a", "you are dead to me!"], padding=True, return_tensors="pt")
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=False))
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=True).shape)