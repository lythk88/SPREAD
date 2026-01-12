import numpy as np
import itertools
import networkx as nx

from more_itertools import powerset
import torch  
from functools import partial
import pickle
from functools import wraps
import time 
from tqdm import tqdm 

from .gsm8k_utils import extract_answer_number, compute_accuracy
from .utils import time_decorator
from .transformers_utils import IntervenableTransformers
__all__ = [
    'shapley_taylor_indices',
    'myerson_interaction_indices',
]

"""
SHAPLEY_TAYLOR_INDICES
"""

def fn_wrapper(T, fn, cpn_dict):
    if T in cpn_dict:
        return 0

    cpns = [T]

    ret = []
    for C in cpns:
        fn_c = fn(frozenset(C))
        ret.append(fn_c)

    cpn_dict[T] = ret
    return 0

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.RandomState()
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def delta_fn(S, T, fn):
    s = len(S)
    T_set = set(T)
    ret = 0

    for W in powerset(S):
        w = len(W)
        value = fn(frozenset(T_set.union(W)))
        ret += (-1)**(w - s) * value

    return ret

def dry_fn_wrapper(T: frozenset, num_nodes: int, mask_dict: dict, mask_list: list):
    if T not in mask_dict:
        mask_list.append(T)
        idx = len(mask_list) - 1
        mask_dict[T] = len(mask_list) - 1
        return idx
    else:
        return mask_dict[T]


def wet_fn_wrapper(T: frozenset, cpn_dict: dict, masked_payoffs: list):
    if T not in cpn_dict:
        raise ValueError(f"{T} is not in cpn_dict")
    else:
        cpn_ids = cpn_dict[T]
        ret = 0
        for idx in cpn_ids:
            ret += masked_payoffs[idx]

    return ret

def shapley_taylor_indices(num_players, fn, ord=2, num_samples=500, random_state=None, return_indices=True):
    rng = check_random_state(random_state)
    indices = np.zeros([num_players]*ord, dtype=np.float64)
    sum_inds = np.zeros_like(indices)
    cnt_inds = np.zeros_like(indices)
    players = np.array(list(range(num_players)))
    # print(players)
    for _ in range(num_samples):
        p = np.array(rng.permutation(num_players))
        inv_p = np.zeros_like(p)
        inv_p[p] = players

        for S in itertools.combinations(players, ord):
            i_k = np.min(inv_p[np.array(S)])
            T = p[:i_k]

            delta = delta_fn(S, T, fn)

            if return_indices:
                for p_S in itertools.permutations(S):
                    sum_inds[p_S] += delta
                    cnt_inds[p_S] += 1

    if return_indices:
        indices = np.divide(sum_inds, cnt_inds, out=np.zeros_like(sum_inds), where=(cnt_inds != 0))

    for r in range(1, ord):
        for S in itertools.combinations(players, r):
            delta = delta_fn(S, tuple(), fn)
            if return_indices:
                # Temporary solution, a not good practice
                # We access the index of a subset S size h by indices[T]
                # where T = (S, S[0], S[0], ...S[0]), |S| = h, |T| = ord
                for i in range(len(S)):
                    p_S = S + (S[i],) * (ord - len(S))
                    indices[p_S] = delta
    return indices

def normalize_reward_func(fn, norm):
    def char_func(data):
        return fn(data) - norm

    return char_func

def get_reward_func(model, 
                    model_inputs,
                    layers_to_prune,
                    golden_answers, 
                    num_attention_groups,
                    num_attention_heads,
                    batch_size=16,
                    reward_type='logprob'):
    
    if num_attention_heads % num_attention_groups != 0:
        raise ValueError("Number of attention heads must be divisible by number of attention groups.")

    def logprob_reward_fn(mask): # fix for all data later
        raise NotImplementedError('Batched logprob reward function not implemented yet')
        mask = list(mask)
        output = model.forward(model_inputs, 
                               heads_to_prune=mask, 
                               layers_to_prune=layers_to_prune)
        return output.loss
    
    def avg_logprob_reward_fn(mask): # fix for all data later
        raise NotImplementedError('Batched logprob reward function not implemented yet')
        mask = list(mask)
        for i in range(0, len(model_inputs['input_ids']), batch_size):
            model_inputs_batch = {k: v[i:i+batch_size] for k, v in model_inputs.items()}
            output = model.forward(model_inputs_batch, 
                                   heads_to_prune=mask, 
                                   layers_to_prune=layers_to_prune)
            loss = output.loss
            
            if i == 0:
                total_loss = loss
            else:
                total_loss += loss
        return total_loss / (len(model_inputs['input_ids']) // batch_size)
    
    
    def accuracy_reward_fn(mask):
        all_heads = list(range(num_attention_groups))
        mask = list(mask)
        if any(element < 0 or element >= num_attention_groups for element in mask):
            raise ValueError("Mask contains invalid group indices.")
        heads_to_keep = []
        num_attention_in_group = num_attention_heads // num_attention_groups
        for element in mask:
            for i in range(num_attention_in_group):
                heads_to_keep.append(element * num_attention_in_group + i)
        model_response_list = []
        for i in range(0, len(model_inputs['input_ids']), batch_size):
            model_inputs_batch = {k: v[i:i+batch_size].to(model.model.device) for k, v in model_inputs.items()}
            heads_to_prune = list(set(all_heads) - set(mask))
            model_responses = model.generate(model_inputs_batch, 
                                   heads_to_prune=heads_to_prune, 
                                   layers_to_prune=layers_to_prune)
            # print(model_responses)
            model_response_list += [extract_answer_number(model_response) for model_response in model_responses]
        accuracy = compute_accuracy(model_response_list, golden_answers)
        return accuracy * 100
    
    
    if reward_type == 'logprob':
        assert 'labels' in model_inputs.keys(), 'labels must be provided for logprob reward'
        if len(model_inputs['input_ids']) > batch_size:
            reward_func = avg_logprob_reward_fn
        else:
            reward_func = logprob_reward_fn
    elif reward_type == 'accuracy':
        f_emp = accuracy_reward_fn([])
        print('empty values:', f_emp)
        # f_emp = accuracy_reward_fn([0,1,2,3,4,5,6,7,8,9,10,11])
        # print(f_emp)
        reward_func = normalize_reward_func(accuracy_reward_fn, f_emp)
    else:
        raise NotImplementedError(f'{reward_type} reward function not implemented yet')
    return reward_func
    
    
def compute_payoffs(reward_func, mask_list):
    # TODO: add support for batched reward_func
    # For now it is sequential
    masked_payoffs = []
    for mask in tqdm(mask_list):
        rewards = reward_func(mask)
        masked_payoffs.append(rewards)
    return masked_payoffs

@time_decorator
def interaction_indices(
    players,
    reward_func,
    ord=1,
    num_samples=100,
    rng=None,
    logger=None,
):
    num_nodes = len(players)

    # dry run
    if logger:
        logger.info('begin dry run...')
    mask_dict = {}
    mask_list = []
    cpn_dict = {}

    par_dry_func = partial(dry_fn_wrapper, num_nodes=num_nodes, mask_dict=mask_dict, mask_list=mask_list)

    dry_func = partial(fn_wrapper,
                        fn=par_dry_func,
                        cpn_dict=cpn_dict)

    # use a fixed seed to make sure dry and wet payoff function will explore the same coalition set
    seed = rng.randint(1e9+7)
    shapley_taylor_indices(num_players=num_nodes,
                            fn=dry_func,
                            ord=ord,
                            num_samples=num_samples,
                            random_state=seed,
                            return_indices=False)

    if logger:
        logger.info('computing payoffs...')
        logger.info(f'Estimating over {len(mask_list)} coalitions')
    
    masked_payoffs = compute_payoffs(reward_func, mask_list)
    if logger:
        logger.info(f"payoffs: {masked_payoffs}")
    wet_func = partial(wet_fn_wrapper, cpn_dict=cpn_dict, masked_payoffs=masked_payoffs)

    if logger:
        logger.info('aggregating results...')
    indices = shapley_taylor_indices(num_players=num_nodes,
                                    fn=wet_func,
                                    ord=ord,
                                    num_samples=num_samples,
                                    random_state=seed)
    # full_payoff = compute_payoffs(reward_func, [list(np.arange(12))])
    # print(full_payoff)
    # print(indices)
    # print(indices.sum())
    return indices
    
if __name__ == '__main__':
    # Example usage
    layers_to_prune = [7]
    model_repo = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    data_path = f"../notebook/tmp/{model_repo.replace('/', '_')}_generated_outputs_1batch.pkl"
    data_path = data_path.replace("1.5B", "7B")
    with open(data_path, "rb") as in_f:
        attention_data_base = pickle.load(in_f)
    
    # attention_data_base['input_ids'] = attention_data_base['input_ids']
    # attention_data_base['labels'] = attention_data_base['labels']
    model = IntervenableTransformers(model_repo, use_auto_model=False)
    
    players = list(np.arange(model.model.config.num_attention_heads))
    
    reward_func = get_reward_func(model, 
                                  attention_data_base, 
                                  layers_to_prune, 
                                  reward_type='logprob')
    
    rng = check_random_state(seed=42)
    
    indices = interaction_indices(
        players,
        reward_func,
        ord=1,
        num_samples=50,
        rng=rng,
    )
    
    stats = model.stats 
    for layer_idx, stats in stats.items():
        # aggregate the mean and variance of the attention output
        means = np.array(stats['mean'])
        variances = np.array(stats['var'])
        print(f"Layer {layer_idx}:")
        print(f"Mean: {np.mean(means, axis=0)}")
        print(f"Variance: {np.mean(variances, axis=0)}")
        print()
    print(indices)