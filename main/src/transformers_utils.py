from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, List, Dict
import json
from pathlib import Path
import logging
import torch 
import transformers 
from functools import partial
import numpy as np
from peft import PeftModel
from accelerate.utils import gather_object
import pickle
import os 
from .hooker import BaseHooker, ZeroOutHooker, WriteAttentionHooker
from .utils import time_decorator
from accelerate.logging import get_logger


class InferenceTransformers:
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='seq_cls',
                 use_auto_model: bool = True,
                 logger=None,
                 use_accelerate: bool = False):
        
        self.model_repo = model_repo
        self.use_accelerate = use_accelerate
        self.build_logger(logger)
        
        self.model_name = model_repo.split('/')[-1]
        self.default_generation_config_path = "src/config/default_generation_config.json"
        if config is not None:
            self.config = config
        else:
            self.config = self.load_config(self.default_generation_config_path)
        
        if use_accelerate:
            from accelerate import PartialState, Accelerator, InitProcessGroupKwargs
            from datetime import timedelta
            import torch.distributed as dist
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.distributed_state = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=5))])
            if not self.distributed_state.state.distributed_type:
                self.distributed_state.init_process_group()
            current_device = torch.cuda.current_device()
            dist.barrier(device_ids=[current_device])
            
            
        self.init_model_tokenizer(model_repo, 
                                lora_path=lora_path,
                                use_auto_model=use_auto_model,
                                task_type=task_type,
                                attention_implementation=self.config.get("attention_implementation", 'eager'),
                                use_accelerate=use_accelerate)

    def build_logger(self, logger):
        if logger is None:
            if not self.use_accelerate:
                self.logger = logging.getLogger(__name__)
                self.logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

                # Add console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

                log_dir = 'default_log'
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(os.path.join(log_dir, f'logfile.log'))
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            else:
                self.logger = get_logger(__name__)
        else:
            self.logger = logger
            
    def init_model_tokenizer(self,
                             model_repo: str,
                             lora_path: str = None, 
                             use_auto_model: bool = True,
                             task_type: str = 'seq_cls',
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
                from .mod_llm.qwen2.modeling_qwen2 import Qwen2ForCausalLM
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
                    
            if 'lama' in model_repo.lower():
                from .mod_llm.llama3.modeling_llama import LlamaForCausalLM
                self.model = LlamaForCausalLM.from_pretrained(model_repo, 
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
                from .mod_llm.gemma2.modeling_gemma2 import GemmaForCausalLM
                self.model = GemmaForCausalLM.from_pretrained(model_repo, 
                                                        torch_dtype=torch.float16, 
                                                        device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                        attn_implementation=attention_implementation)
            else:
                raise ValueError(f"Model {model_repo} not supported. or set use_auto_model=True instead")
        
        if self.logger:
            self.logger.info(f"Model loaded from {model_repo if lora_path is None else lora_path}")
        else:
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
    def load_config(filepath: str = "src/config/default_generation_config.json") -> Dict:
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
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        seq_len = model_inputs['input_ids'].shape[1]
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
        )
        
        if return_raw_output:
            return output
        else:
            response = self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True)
            return response
        
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
    
class AttentionWriterTransformers(InferenceTransformers):
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='seq_cls',
                 use_auto_model: bool = False,
                 attention_path: str = "tmp",
                 logger=None):
        
        super().__init__(model_repo, 
                         config, 
                         lora_path, 
                         task_type, 
                         use_auto_model,
                         logger=logger)
        self.attention_path = attention_path
        if lora_path is None:
            self.attention_path = os.path.join(self.attention_path, model_repo.replace('/', '_'))
        else:
            lora_names = '_'.join(lora_path.split('/')[-5:])
            self.attention_path = os.path.join(self.attention_path, lora_names)
            
        self.attention_hooker = WriteAttentionHooker(
            layer_list=list(np.arange(self.model.config.num_hidden_layers)), 
            stat_track=False, 
            attention_path=self.attention_path)
    
    @torch.no_grad()
    def forward(self, inputs: Union[str, List[str], Dict]) -> str:
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        output = self.model(**model_inputs, 
                            edit_fn=self.attention_hooker,
                            use_cache=False)
        # self.attention_hooker.__call__.print_calls()
        # self.attention_hooker.__call__.reset_count()
        # self.attention_hooker.write_attention()
        return output
    

class IntervenableTransformers(InferenceTransformers):
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='seq_cls',
                 use_auto_model: bool = True,
                 logger=None,
                 use_accelerate: bool = False):
        super().__init__(model_repo, 
                         config, 
                         lora_path, 
                         task_type, 
                         use_auto_model, 
                         logger=logger,
                         use_accelerate=use_accelerate)
    
    def cache_outputs(self, outputs, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as out_f:
            pickle.dump(outputs, out_f)
            
    def load_cached_prompts(self, cache_path):
        with open(cache_path, "rb") as in_f:
            return pickle.load(in_f)
        
    # TODO: bad practice, fix this into another function
    @torch.no_grad() 
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        return_raw_output: bool = False,
        heads_to_prune: List[int] = [3],
        layers_to_prune: List[int] = [3],
        stat_track: bool = True,
        save_every_n_gens: int = 10,
        prompt_cache_path: str = "tmp_attention",
        use_prompt_cache: bool = True,
    ):

        if config is None:
            config = self.config
        
            
        attention_hooker = ZeroOutHooker(head_indices=heads_to_prune, 
                                         layer_list=layers_to_prune, 
                                         stat_track=stat_track,
                                         logger=self.logger)

        
        if not self.use_accelerate: 
            if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
                model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
            elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
                model_inputs = inputs
            elif isinstance(inputs, str):
                model_inputs = self.tokenizer(inputs, return_tensors='pt')
            else:
                raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
            seq_len = model_inputs['input_ids'].shape[1]
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            outputs = self.model.generate(
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
                edit_fn=attention_hooker
            )
            attention_hooker.__call__.print_calls()
            attention_hooker.log_stats()
            attention_hooker.__call__.reset_count()
            self.stats = attention_hooker.get_stats()
            if return_raw_output:
                return outputs
            else:
                response = self.tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)
                return response
        else:  # TODO: BAD PRACTICE, FIX THIS # https://github.com/huggingface/accelerate/issues/2733
            
            num_return_sequences = config.get("num_return_sequences", 1)
            if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
                batch_size_per_device = config.get('batch_size_per_device')
                formatted_prompts = [inputs[i : i + batch_size_per_device] for i in range(0, len(inputs), batch_size_per_device)]
                model_inputs =[self.tokenizer(formatted_prompt, padding=True, return_tensors='pt')
                               for formatted_prompt in formatted_prompts]
                save_every_n_gens_ = save_every_n_gens * self.distributed_state.num_processes
                split_model_inputs = [model_inputs[i : i + save_every_n_gens_] for i in range(0, len(model_inputs), save_every_n_gens_)]
                print([len(x) for x in split_model_inputs])
            else:
                raise NotImplementedError(f"Invalid input type {type(inputs)}. Must be list of str.")
            
            
                
            if self.distributed_state.is_main_process:
                print()
                print(f"save_every_n_gens: {save_every_n_gens} | len model_inputs: {len(model_inputs)} (should be larger than sene {save_every_n_gens})")
                print(f"len inputs: {len(inputs)}")
                print(f"len split_model_inputs: {len(split_model_inputs)}")
                print(f"prompt cache path: {prompt_cache_path}")
            outputs = []
            for split_count, splitted_model_inputs in enumerate(split_model_inputs):
                if self.distributed_state.is_main_process:
                    print(f"\nsplit_count: {split_count} / len split_model_inputs: {len(splitted_model_inputs)}")
                splitted_model_inputs_cache_path = os.path.join(prompt_cache_path, f"{split_count}_cached_outputs.pkl")
                if os.path.exists(splitted_model_inputs_cache_path) and use_prompt_cache:
                    prompt_cache = self.load_cached_prompts(splitted_model_inputs_cache_path)
                    outputs = prompt_cache.copy()
                    print(f"{splitted_model_inputs_cache_path} exists, loading data from cache. len outputs: {len(outputs)}. Proceed with next generation")
                    self.distributed_state.state.wait_for_everyone()
                    continue
                
                self.distributed_state.state.wait_for_everyone()
                with self.distributed_state.split_between_processes(splitted_model_inputs) as batched_prompts:
                    generated_texts_across_device = []
                    for batch in batched_prompts:
                        torch.cuda.synchronize()
                        batch = {k: v.to(self.distributed_state.device) for k, v in batch.items()}
                        seq_len = batch['input_ids'].shape[1]
                        print(f'process no. {self.distributed_state.process_index} | seq_len: {seq_len} | batch: {batch["input_ids"].shape}')
                        output = self.model.generate(
                            **batch,
                            max_new_tokens=config.get("max_new_tokens", 50),
                            temperature=config.get("temperature", 1.0),
                            top_k=config.get("top_k", 50),
                            top_p=config.get("top_p", 1.0),
                            repetition_penalty=config.get("repetition_penalty", 1.0),
                            num_return_sequences=num_return_sequences,
                            do_sample=config.get("do_sample", True),
                            use_cache=config.get("use_cache", True),
                            pad_token_id=self.tokenizer.eos_token_id,
                            edit_fn=attention_hooker
                        )
                        generated_texts_across_device.extend(self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True))
                        if self.distributed_state.is_main_process:
                            print(f"len generated_texts_across_device: {len(generated_texts_across_device)} | len batched_prompts: {len(batched_prompts)}  current batch: {batch['input_ids'].shape}")
                    
                    generated_texts_across_device = gather_object(generated_texts_across_device)
                    if self.distributed_state.is_main_process:
                        outputs.extend(generated_texts_across_device)
                        
                self.distributed_state.state.wait_for_everyone()
                if self.distributed_state.is_main_process:
                    print(f"Caching {len(outputs)} outputs to {splitted_model_inputs_cache_path}")
                    self.cache_outputs(outputs, splitted_model_inputs_cache_path)
                self.distributed_state.state.wait_for_everyone()

            if self.distributed_state.is_main_process:
                print(f'len outputs ({len(outputs)}) vs len inputs ({len(inputs)})')
                
                if num_return_sequences == 1:
                    outputs = outputs[:len(inputs)]
                    assert len(outputs) == len(inputs), f"Length mismatch between inputs and outputs: {len(inputs)} != {len(outputs)}"    
                    return outputs
                else:
                    outputs = outputs[:len(inputs) * num_return_sequences]
                    assert len(outputs) == len(inputs) * num_return_sequences, f"Length mismatch between inputs and outputs: {len(inputs)} != {len(outputs)}"    
                    return outputs
                
    
    # {'no_prune_Precalculus_acc_mean': 0.53125, 'no_prune_Precalculus_acc_std': 0.4990224819584785}
    # {'no_prune_Intermediate Algebra_acc_mean': 0.59375, 'no_prune_Intermediate Algebra_acc_std': 0.4911323014219285}
    
    @torch.no_grad()
    def forward(
        self,
        inputs: Union[str, List[str], Dict],
        heads_to_prune: List[int] = [3],
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
        
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        attention_hooker = ZeroOutHooker(head_indices=heads_to_prune, 
                                         layer_list=layers_to_prune,
                                         stat_track=stat_track,
                                         logger=self.logger)
        
        output = self.model(**model_inputs, 
                            edit_fn=attention_hooker,
                            use_cache=False)
        output.logits = output.logits.to('cpu')
        attention_hooker.__call__.print_calls()
        attention_hooker.__call__.reset_count()
        self.stats = attention_hooker.get_stats()

        # attention_hooker.log_stats()
        return output
    
    

if __name__ == '__main__':
    # log_folder = '../log'
    # os.makedirs(log_folder, exist_ok=True)
    # log_file = os.path.join(log_folder, 'transformers_utils.log')
    # logging.basicConfig(
    #     filename=log_file,
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s"
    # )
    
    os.makedirs("tmp", exist_ok=True)
    model_repo = "Qwen/Qwen2.5-Math-7B-Instruct"
    inference_transformers = IntervenableTransformers(model_repo, 
                                                   use_auto_model=False)
    breakpoint()
    data_path = f"notebook/tmp/{model_repo.replace('/', '_')}_generated_outputs_1batch.pkl"
    # data_path = data_path.replace("1.5B", "7B")
    with open(data_path, "rb") as in_f:
        attention_data_base = pickle.load(in_f)
    
    attention_data_base['input_ids'] = attention_data_base['input_ids']
    attention_data_base['labels'] = attention_data_base['labels']
    default_config = InferenceTransformers.load_config('config/default_generation_config.json')
    
    model_inputs = inference_transformers.generate(attention_data_base, 
                                                   return_raw_output=False,
                                                   heads_to_prune=[],
                                                   layers_to_prune=[],)
    print(model_inputs)
    from grading import grader 
    
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