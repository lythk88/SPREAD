import numpy as np
from datasets import load_dataset 
import re 
from typing import Union, List, Dict
import math 
from collections import defaultdict

ANSWER_PROMPT = "The answer is: "
QUESTION_PROMPT = "\nLet's think step by step\n"

qwen_chat_math_prompt_template = """<|im_start|>system
Solve the following question.<|im_end|>\n
<|im_start|>user
{question}\n<|im_end|>
<|im_start|>assistant\n"""

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)

def get_gsm8k_dataset(split='test'): # split can be train
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset[split]

    question = [f"{example['question']}\n" for example in test_set]
    answer = []
    # get numerical answer
    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '') 
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)
    return question, answer

def apply_prompt(sample, model_repo):
    if 'qwen' in model_repo.lower():
        if 'instruct' in model_repo.lower():
            prompt = qwen_chat_math_prompt_template.format(question=sample)
        else:
            prompt = f"Question: {sample['problem']}. Let's think step by step."
    
    return prompt 

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit

def is_correct(completion, answer):
    gold = answer
    # gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        # return math.isclose(float(answer), float(pred), rel_tol=0, abs_tol=1e-4)
        try:
            return math.isclose(float(answer), float(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))

class GSM8kEvaluator:
    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {'mean': [], 'var': []})
    
    def get_stats(self):
        print('average success rate:', np.mean(self.stats['success_rate']['mean']))
        print('average success rate variance:', np.mean(self.stats['success_rate']['var']))
        return self.stats
    
    def get_success_rate(self, model_output, ground_truth_answers, num_return_sequences, stat_track=False):
        assert len(model_output) % num_return_sequences == 0, "Output size must be divisible by num_return_sequences."
        assert len(ground_truth_answers) == len(model_output) // num_return_sequences, f"Ground truth size ({len(ground_truth_answers)}) must match model output size ({len(model_output)}) divided by num_return_sequences ({num_return_sequences})."

        total_questions = len(ground_truth_answers)
        per_question_success_rates = []
        for i in range(total_questions):
            model_answers = model_output[i * num_return_sequences:(i + 1) * num_return_sequences]
            ground_truth_answer = ground_truth_answers[i]
            success_count = sum(
                is_correct(model_answer, ground_truth_answer) for model_answer in model_answers
            )
            per_question_success_rate = success_count / num_return_sequences
            per_question_success_rates.append(per_question_success_rate)

        if stat_track:
            self.stats['success_rate']['mean'].extend(per_question_success_rates)
            self.stats['success_rate']['var'].extend(per_question_success_rates)
            
        return per_question_success_rates, np.mean(per_question_success_rates), np.std(per_question_success_rates)
    
if __name__ == "__main__":
    train_question, train_answer = get_gsm8k_dataset(split='train')
    test_question, test_answer = get_gsm8k_dataset(split='test')
    breakpoint()