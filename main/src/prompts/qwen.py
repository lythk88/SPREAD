MATH_PROMPT_TEMPLATE = \
    "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
    "<|im_start|>user\n{input}<|im_end|>\n" \
    "<|im_start|>assistant\n"

BASE_PROMPT_TEMPLATE = \
    "Question: {input}\nAnswer:\\boxed{{}}\n" 
    
MCQ_PROMPT_TEMPLATE = \
    "Question: {input}\nAnswer Choices: (A)\n" 
