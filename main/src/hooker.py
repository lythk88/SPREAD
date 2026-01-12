import os 
from typing import Union, List, Dict
import torch 
import numpy as np
from collections import defaultdict
from .utils import count_decorator

class BaseHooker:
    def __init__(self, 
                 layer_list: List[int], 
                 stat_track: bool = True,
                 logger=None) -> None:
        self.attention = None
        self.layer_list = layer_list
        self.stats: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {'mean': [], 'var': [], 'norm': []})
        self.current_layer = None
        self.stat_track = stat_track
        self.logger = logger
        
    def log_stats(self):
        if not self.stat_track:
            self.logger.info("Stat tracking is disabled.")
            return
        
        for layer_idx, stats in self.stats.items():
            # aggregate the mean and variance of the attention output
            means = np.array(stats['mean'])
            variances = np.array(stats['var'])
            self.logger.debug(f"Layer {layer_idx}:")
            self.logger.debug(f"Mean: {np.mean(means, axis=0)}")
            self.logger.debug(f"Variance: {np.mean(variances, axis=0)}")
            self.logger.debug(f"Norm: {np.mean(stats['norm'], axis=0)}\n")
            
    def get_stats(self):
        return self.stats
    
    def track_stats(self, attn_output):
        mean = attn_output.mean(dim=(0, 1, 3)).detach().cpu().float().numpy()
        var = attn_output.var(dim=(0, 1, 3)).detach().cpu().float().numpy()
        norm = attn_output.norm(dim=3).norm(dim=1).norm(dim=0).detach().cpu().float().numpy()
        self.stats[self.current_layer]['mean'].append(mean)
        self.stats[self.current_layer]['var'].append(var)
        self.stats[self.current_layer]['norm'].append(norm)
        
    @count_decorator
    def __call__(self, attn_output):
        if self.current_layer not in self.layer_list:
            return attn_output
        
        if self.stat_track:
            self.track_stats(attn_output)
        
        self.attention = attn_output.detach().cpu()
        return attn_output
    
class WriteAttentionHooker(BaseHooker):
    def __init__(self, layer_list: List[int], 
                 stat_track: bool = True, 
                 attention_path: str = "tmp",
                 save_last_token_only: bool = True,
                 logger=None) -> None:
        
        super().__init__(layer_list, stat_track, logger)
        self.target_attention_names = ["pre_o_proj", "post_o_proj"]
        self.attention_path = attention_path 
        self.attention_dict = {}
        self.save_lt_only = save_last_token_only
    
    def write_attention(self):
        if self.attention_dict is None:
            raise ValueError("No attention to write.")
        
        if not os.path.exists(self.attention_path):
            os.makedirs(self.attention_path)
        
        for layer_name, attn in self.attention_dict.items():
            # try:
            stacked_attn = np.concatenate(attn, axis=0)
            # except:
            #     print(attn[-2].shape)
            #     print(attn[-1].shape)
            #     print(self.current_layer)
            #     import sys 
            #     sys.exit(1)
            attn_path = os.path.join(self.attention_path, f"{layer_name}.npy")
            np.save(attn_path, stacked_attn)
    
    @count_decorator
    def __call__(self, attn_output, attention_name="post_o_proj"):
        if attention_name not in self.target_attention_names:
            return attn_output
        
        if self.current_layer not in self.layer_list:
            return attn_output
        
        if not isinstance(attn_output, torch.Tensor):
            raise TypeError("attn_output must be a torch.Tensor")
       
        if self.stat_track:
            self.track_stats(attn_output)
        
        # batch_size, seq_len, num_attention_head, head_dim = attn_output.shape
        # if any(idx < 0 or idx >= num_attention_head for idx in self.head_indices):
        #     raise ValueError("head_indices contains invalid head indices")
        layer = f'{self.current_layer}_{attention_name}'
        if layer not in self.attention_dict:
            self.attention_dict[layer] = []
        
        if self.save_lt_only:
            save_attn_output = attn_output[:, -1, ...].detach().cpu().float().numpy()
        else:
            save_attn_output = attn_output.detach().cpu().float().numpy()
        self.attention_dict[layer].append(save_attn_output)
        return attn_output
    
    
class ZeroOutHooker(BaseHooker):
    def __init__(self, 
                 head_indices: List[int], 
                 layer_list: List[int], 
                 stat_track: bool = True, 
                 logger=None) -> None:
        super().__init__(layer_list, stat_track, logger)
        self.head_indices = head_indices
        self.target_attention_names = ["pre_o_proj"]
        
    
    @count_decorator
    def __call__(self, attn_output, attention_name="post_o_proj"):
        if attention_name not in self.target_attention_names:
            return attn_output
        
        if self.current_layer not in self.layer_list:
            return attn_output
        
        if not isinstance(attn_output, torch.Tensor):
            raise TypeError("attn_output must be a torch.Tensor")
        if len(attn_output.shape) != 4:
            raise ValueError("attn_output must have shape (batch_size, seq_len, num_attention_head, head_dim)")
        if not all(isinstance(idx, (int, np.integer)) for idx in self.head_indices):
            raise ValueError("head_indices must be a list of integers")
       
        if self.stat_track:
            self.track_stats(attn_output)
            
        batch_size, seq_len, num_attention_head, head_dim = attn_output.shape
        if any(idx < 0 or idx >= num_attention_head for idx in self.head_indices):
            raise ValueError("head_indices contains invalid head indices")

        mask = torch.ones(num_attention_head, dtype=attn_output.dtype, device=attn_output.device)
        mask[self.head_indices] = 0  
        mask = mask.view(1, 1, num_attention_head, 1)  
        attn_output = attn_output * mask  
        return attn_output