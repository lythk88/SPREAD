import os
import sys
import argparse
import pickle
import torch
import time
import numpy as np
import logging
import re
import random
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to import properly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_jsonl
from src.inference_utils import InferenceEngine, CacheManager
from src.prompts.qwen import MATH_PROMPT_TEMPLATE, BASE_PROMPT_TEMPLATE
from src.evaluation.parser import extract_answer, strip_string, parse_ground_truth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command-line arguments for Baseline experiments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Baseline generation.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Size of the batch to process (default: 4)"
    )
    parser.add_argument(
        "--input_start",
        type=int,
        default=0,
        help="Start of the input range (default: 0)"
    )
    parser.add_argument(
        "--input_end",
        type=int,
        default=100,
        help="End of the input range (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Model repository to use"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun inference ignoring cache"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory for cache files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/algorithm1_steering",
        help="Directory for results"
    )
    parser.add_argument(
        "--run_name_before",
        type=str,
        default=None,
        help="Custom name for the run (default: auto-generated from parameters)"
    )
    parser.add_argument(
        "--number_candidate",
        type=int,
        default=5,
        help="Number of candidates to generate per problem (default: 5)"
    )
    parser.add_argument(
        "--save_all_candidates",
        action="store_true",
        help="Save all candidate responses, not just the best one"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/math/test.jsonl",
        help="Path to dataset file (default: data/math/test.jsonl for PRM800k). Other options: data/gsm8k/test.jsonl, data/aime24/test.jsonl"
    )
    return parser.parse_args()

def extract_candidate_answer(response, dataset="math"):
    """
    Extract the answer from a candidate response.

    Args:
        response: Text response from the model
        dataset: Dataset type (math, gsm8k, etc.)

    Returns:
        Extracted answer string
    """
    try:
        # First try to extract with the standard extraction function
        answer = extract_answer(response, data_name=dataset)
        if answer and answer.strip():
            return strip_string(answer)

        # If that didn't work, try more aggressive methods for math problems
        if "Answer:" in response or "answer:" in response:
            # Find answer after "Answer:" tag
            if "Answer:" in response:
                parts = response.split("Answer:")
            else:
                parts = response.split("answer:")

            answer_text = parts[-1].strip()

            # Get the first sentence of the answer
            if "." in answer_text:
                answer_text = answer_text.split(".")[0].strip()

            # Take only the final number if it seems to be a calculated result
            number_match = re.search(r'-?\d*\.?\d+', answer_text)
            if number_match:
                return number_match.group(0)

            return strip_string(answer_text)

        # Try to find the last number in the text as a last resort for math problems
        number_matches = re.findall(r'-?\d*\.?\d+', response)
        if number_matches:
            return number_matches[-1]

        return response.strip()

    except Exception as e:
        logger.warning(f"Error extracting answer: {e}")
        # Last resort fallback
        return response.strip()
    
def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(args):
    """Run steering algorithm1 inference with the specified parameters."""
    # Load data from specified path
    project_root = str(Path(__file__).parent.parent.parent)
    data_path = os.path.join(project_root, args.data_path)
    logger.info(f"Loading data from: {data_path}")

    # Identify dataset type
    if 'gsm8k' in args.data_path:
        dataset_name = 'gsm8k'
    elif 'aime24' in args.data_path:
        dataset_name = 'aime24'
    elif 'math' in args.data_path:
        dataset_name = 'math' 
    elif 'olympiadbench' in args.data_path:
        dataset_name = 'olympiadbench'

    logger.info(f"Dataset type: {dataset_name}")
    data = list(load_jsonl(data_path))[args.input_start:args.input_end]

    # Create directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model_repo = args.model_repo
    model_name = model_repo.split('/')[-1].lower()

    logger.info(f"Initializing InferenceEngine with {model_repo}...")
    engine = InferenceEngine(model_repo, use_auto_model=True)
    
    # Format prompts
    if engine.model_repo in "Qwen/Qwen2.5-Math-1.5B-Instruct":
        prompts = [MATH_PROMPT_TEMPLATE.format(input=question['problem']) for question in data]
    elif engine.model_repo in "Qwen/Qwen2.5-1.5B":
        prompts = [BASE_PROMPT_TEMPLATE.format(input=question['problem']) for question in data]
    # print("First prompt: ", prompts[0])


    # Create run name and cache path
    run_name = args.run_name_before or f"algorithm1_steering_n{args.number_candidate}_temp{args.temperature}_{args.input_start}_{args.input_end}"
    cache_path = f"{args.cache_dir}/{dataset_name}_{run_name}.pkl"

    # Create cache manager
    cache_manager = CacheManager(inference_engine=engine, cache_file_path=cache_path, batch_size=args.batch_size)

    # Configure engine for temperature sampling with multiple candidates
    engine.config["temperature"] = args.temperature
    engine.config["do_sample"] = args.temperature > 0
    engine.config["num_return_sequences"] = 1

    engine.config["return_dict_in_generate"] = True
    engine.config["output_scores"] = True

    logger.info(f"Running N-{args.number_candidate} inference on {len(prompts)} problems with temperature {args.temperature}...")
    

    results = {
        "metadata": {
            "model": model_repo,
            "temperature": args.temperature,
            "input_start": args.input_start,
            "input_end": args.input_end,
            "generation_config": engine.config,
            "total_wallclock_time": 0,  # Will be updated at the end
            "num_problems": len(data),  # Store number of problems for per-prompt time calculation
            "total_tokens_generated": 0,  # Will track total tokens generated
            "total_prompt_tokens": 0,    # Will track total prompt tokens
            "total_candidates": 0,       # Will track total number of candidates
            "total_output_tokens": 0     # Will track total output tokens (for compatible with other metrics)
        },
        "results": []
    }

    # For tracking total wallclock time
    start_wallclock_time = time.time()
    generations = []

    # print("Engine config: ", engine.config)
    engine.config['num_hidden_layers'] = engine.model.config.num_hidden_layers
    # print("Engine config: ", engine.config)
    # Process each problem separately
    for i, prompt in enumerate(tqdm(prompts)):   
        prompt_inputs = engine.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate output with steering
        output = engine.generate(
            prompt_inputs,
            config={"max_new_tokens": 2048, "temperature": args.temperature, "do_sample": True, "num_return_sequences": args.number_candidate},
            return_raw_output=False
        )
    
        # Store result
        steered_outputs = output[0]["text"]  # output["text"] is a list of strings
        generations.append(steered_outputs)
        for x in steered_outputs: 
            print(repr(x[-50:]))

        # Track token counts from the generation
        input_tokens = 0
        output_tokens = 0

        # Track the number of candidates we generated for this problem
        candidate_count = len(steered_outputs)
        results["metadata"]["total_candidates"] += candidate_count

        # Get token counts if available in the generation output
        if isinstance(generations, dict):
            if "input_token_count" in generations:
                input_tokens = generations["input_token_count"]
                results["metadata"]["total_prompt_tokens"] += input_tokens

            if "output_token_count" in generations:
                output_tokens = generations["output_token_count"]
                results["metadata"]["total_tokens_generated"] += output_tokens
                results["metadata"]["total_output_tokens"] += output_tokens

            # Alternative counting method if specific counts aren't available
            elif "token_count" in generations:
                # For best-of-N, we typically generate N times more tokens
                gen_tokens = generations["token_count"]
                # Estimate input vs output tokens
                if args.number_candidate > 1:
                    # With multiple sequences, we're reusing the prompt
                    output_tokens = gen_tokens - input_tokens
                    results["metadata"]["total_tokens_generated"] += output_tokens
                    results["metadata"]["total_output_tokens"] += output_tokens

        # Create result entry
        result = {
            'problem': data[i]['problem'],
            'ground_truth': data[i]['answer'][0] if isinstance(data[i]['answer'], list) else data[i]['answer'],
            'input_tokens': input_tokens,
            'output_tokens': output_tokens            
        }

        # Optional save all candidate
        if args.save_all_candidates:
            result['responses'] = steered_outputs
            
        results["results"].append(result)

    # Calculate total wallclock time
    end_wallclock_time = time.time()
    total_wallclock_time = end_wallclock_time - start_wallclock_time

    # Update metadata with timing information
    results["metadata"]["total_wallclock_time"] = total_wallclock_time
    
    # Calculate average output tokens per candidate (for fairer comparison)
    if results["metadata"].get("total_candidates", 0) > 0:
        results["metadata"]["avg_output_tokens_per_candidate"] = results["metadata"]["total_output_tokens"] / results["metadata"]["total_candidates"]
    else:
        results["metadata"]["avg_output_tokens_per_candidate"] = 0
    
    # Traditional average output tokens per problem (kept for backward compatibility)
    if len(data) > 0:
        results["metadata"]["avg_output_tokens"] = results["metadata"]["total_output_tokens"] / len(data)

    # Save raw results
    results_file = f"{args.output_dir}/{run_name}_raw.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Raw results saved to {results_file}")

    # Create a log file
    log_file = f"{args.output_dir}/logs.txt"
    with open(log_file, 'a') as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: {model_repo}\n")
        f.write(f"Number of candidate: {args.number_candidate}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Input range: {args.input_start}-{args.input_end}\n")
        f.write(f"Problems: {len(data)}\n")
        f.write(f"Total wallclock time: {total_wallclock_time:.2f} seconds\n")
        f.write(f"Average time per problem: {total_wallclock_time/len(data):.2f} seconds\n")
        f.write(f"Total prompt tokens: {results['metadata']['total_prompt_tokens']}\n")
        f.write(f"Total generated tokens: {results['metadata']['total_tokens_generated']}\n")
        f.write(f"Total candidates: {results['metadata']['total_candidates']}\n")
        f.write(f"Average tokens per problem: {results['metadata']['total_output_tokens']/len(data):.2f}\n")
        f.write(f"Average tokens per candidate: {results['metadata']['avg_output_tokens_per_candidate']:.2f}\n")
        f.write(f"Raw results saved to: {results_file}\n")
        f.write(f"To evaluate: python evaluate_strategies.py --input {results_file} --plot\n\n")

    logger.info(f"Run information saved to log file: {log_file}")
    logger.info(f"Total wallclock time: {total_wallclock_time:.2f} seconds")
    logger.info(f"Average time per problem: {total_wallclock_time/len(data):.2f} seconds")
    logger.info(f"Total prompt tokens: {results['metadata']['total_prompt_tokens']}")
    logger.info(f"Total generated tokens: {results['metadata']['total_tokens_generated']}")
    logger.info(f"Total candidates: {results['metadata']['total_candidates']}")
    logger.info(f"Average tokens per problem: {results['metadata']['total_output_tokens']/len(data):.2f}")
    logger.info(f"Average tokens per candidate: {results['metadata']['avg_output_tokens_per_candidate']:.2f}")
    logger.info(f"To evaluate results, run: python evaluate_strategies.py --input {results_file} --plot")
    
    return results

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    results = run(args)