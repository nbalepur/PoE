# =========================================== Parameter Setup ===========================================

# hyperparameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    '-m',
    type=str,
    help="(Nick)name of the model in directory",
    default="llama 7b",
)
parser.add_argument(
    "--model_name_hf",
    type=str,
    help="Name of the model on hugging face",
    default="meta-llama/Llama-2-7b-hf",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of the dataset",
    default='commonsense_qa',
)
parser.add_argument(
    "--use_cot",
    type=str,
    help="Should chain of thought prompting be used?",
    default="False",
)
parser.add_argument(
    "--answer_type",
    type=str,
    help="Should we identify correct or incorrect answers?",
    default="correct",
)
parser.add_argument(
    "--load_in_4bit",
    type=str,
    help="Should we load the model in 4 bit?",
    default="False",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    help="Where to store the model cache",
    default="",
)
parser.add_argument(
    "--out_dir",
    type=str,
    help="Where to save the outputs",
    default="",
)
parser.add_argument(
    "--hf_token",
    type=str,
    help="Huggingface token for access to the model",
    default="",
)

args = parser.parse_args()
print(args)
use_cot = (args.use_cot == 'True')
load_in_4bit = (args.load_in_4bit == 'True')
aim_for_correct = (args.answer_type == 'correct')

dataset_name = args.dataset_name
model_name = args.model_name
hf_model_name = args.model_name_hf
cot_normal_str = "chain_of_thought" if use_cot else "normal"

# imports and directory setup
import pickle
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
import copy
from transformers import AutoTokenizer
from huggingface_hub.hf_api import HfFolder
os.environ['TRANSFORMERS_CACHE'] = args.cache_dir

hf_token = args.hf_token
HfFolder.save_token(hf_token)


INSTRUCTIONS = f'Your goal is to identify the {"correct" if aim_for_correct else "incorrect"} answer to the multiple-choice question.'

prompt_dir = f'./prompts/{dataset_name}'
results_dir = args.cache_dir
model_name = hf_model_name

# =========================================== Load Model ===========================================

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir = args.cache_dir)

# set up pipeline
stop_token = '\nQuestion:'
dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "auto": "auto",
}['auto']
pipe = pipeline(
    model=hf_model_name,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=dtype,
    min_new_tokens=5,
    max_new_tokens=1000,
    model_kwargs={"cache_dir": args.cache_dir, "temperature": 0.3, "do_sample": True, "load_in_4bit": load_in_4bit}
)
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens = [], prompt_len = 0):
      super().__init__()
      self.prompt_len = prompt_len
      self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      sublist = self.stop_tokens
      input_ids = input_ids[0].tolist()
      seq_in_gen = sublist in [input_ids[i:len(sublist)+i] for i in range(self.prompt_len, len(input_ids))]
      return seq_in_gen

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:], prompt_len=input_ids.shape[1])])
    return pipe(prompt, 
    stopping_criteria=stopping_criteria,
    temperature=0.3,
    return_full_text=False)[0]['generated_text'][:-len(stop_token)].strip()

# =========================================== Load Prompts ===========================================

f = open(prompt_dir + f'/{"correct" if aim_for_correct else "incorrect"}_answer_{"cot_" if use_cot else ""}prompt.txt', 'r')
prompt_prefix = INSTRUCTIONS + '\n===\n' + ''.join(f.readlines())

with open(f'./data/{dataset_name}/normal_question_2_choices.pkl', 'rb') as handle:
    ds = pickle.load(handle)

import numpy as np
def convert_text_to_answer(txt, n):
    idxs = [txt.find('(' + str(chr(ord('A') + i)) + ')') for i in range(n)]
    if sum(idxs) == -1 * n:
        return None, None
    idxs = [100000 if i == -1 else i for i in idxs]
    return np.argmin(idxs), chr(ord('A') + np.argmin(idxs))

# =========================================== Run Generation ===========================================

answers = {'index': [], 'letter': [], 'text': [], 'raw_text': [], 'prompt': []}
for i in tqdm.tqdm(range(len(ds['questions']))):

    # create prompt
    prompt = prompt_prefix
    q, c = ds['questions'][i].replace('  ', ' '), ds['choices'][i]
    prompt += f'Question: {q}\n'
    prompt += 'Choices:\n'
    for j in range(len(c)):
        l, t = chr(ord('A') + j), c[j]
        prompt += f'({l}) {t}\n'

    if aim_for_correct:
        prompt += "Correct Answer:"
    else:
        prompt += "Incorrect Answer:"
    prompt = prompt.replace('===\n', '\n')

    # generate answer
    out_text = generate_text(prompt)

    idx, ltr = convert_text_to_answer(out_text, len(c))

    # append to output
    if idx != None:
        answer = c[idx]
        answers['text'].append(answer)
    else:
        answers['text'].append(None)
    answers['index'].append(idx)
    answers['letter'].append(ltr)
    answers['raw_text'].append(out_text)
    answers['prompt'].append(prompt)

# =========================================== Save Results ===========================================

import pickle
with open(f'{results_dir}/{"correct" if aim_for_correct else "incorrect"}_answer{"_cot" if use_cot else ""}.pkl', 'wb') as handle:
    pickle.dump(answers, handle, protocol=pickle.HIGHEST_PROTOCOL)