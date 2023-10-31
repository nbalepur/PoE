# =========================================== Parameter Setup ===========================================

# hyperparameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    '-m',
    type=str,
    help="(Nick)name of the model in directory",
    default="gpt 3.5",
)
parser.add_argument(
    "--model_name_gpt",
    type=str,
    help="Name of the GPT model to use for the API",
    default="gpt-3.5-turbo",
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
    "--open_ai_key",
    type=str,
    help="OpenAI API key for GPT models",
    default="",
)
parser.add_argument(
    "--out_dir",
    type=str,
    help="Where to save the outputs",
    default="",
)

args = parser.parse_args()
use_cot = (args.use_cot == 'True')
aim_for_correct = (args.answer_type == 'correct')

dataset_name = args.dataset_name
model_name = args.model_name
gpt_model_name = args.model_name_gpt
cot_normal_str = "chain_of_thought" if use_cot else "normal"

load_in_4bit = ('180' in model_name)

# imports and directory setup
import pickle
import tqdm
import os
import time
import copy


INSTRUCTIONS = f'Your goal is to identify the {"correct" if aim_for_correct else "incorrect"} answer to the multiple-choice question.'

prompt_dir = f'./prompts/{dataset_name}'
results_dir = args.out_dir
model_name = gpt_model_name

# =========================================== Load Model ===========================================


import openai
openai.api_key = args.open_ai_key

# set up pipeline
stop_token = 'Question:'

def generate_text(prompt, time_wait):
    time.sleep(2**(time_wait))
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": prompt
                },
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[stop_token]
        )
        return response['choices'][0]['message']['content']
    except:
        return generate_text(prompt, time_wait+1)
    

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
    out_text = generate_text(prompt, 0)

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