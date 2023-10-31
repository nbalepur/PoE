# Process of Elimination (POE)

This repository is the official implementation of "Knowing What's Wrong and What's Right: Evaluating Process of Elimination in Large Language Models", soon to be uploaded to Arxiv.

<p align="center">
  <img src="/images/POE_Intro.png"></img>
</p>

## Overview

This repository contains the code and dataset to run the direct answer and process of elimination strategies, with and without chain of thought, on our four tested commonsense reasoning and scientific reasoning datasets

## Setup

Python 3.10.0 and pip 23.2.1 were used when running the code in this repository. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:
```
pip install -r requirements.txt 
```

The most important files in this repository are as follows:
* `/data/`: Contains the 2-choice multiple-choice QA datasets
* `/prompts/`: Contains the prompts used in our experiments
* `/model/`: Contains the inference code for running LLMs, either through the OpenAI API or Huggingface

## Usage

You can run inference on the Huggingface models with the following command: 
```
python ./model/answer_2_choices_hf.py
```
The script requires the following arguments:
* `model_name`: (Nick)name of the model for savin the results. String type
* `model_name_hf`: Name of the model on huggingface. String type
* `dataset_name`: Name of the dataset in `/data/`. String type
* `use_cot`: Should chain of thought prompting be used? String type ("True" for COT, "False" for Base)
* `answer_type`: "Should we identify correct or incorrect answers? String type ("correct" for DA Strategy, "incorrect" for POE Strategy)
* `load_in_4bit`: Should we load the model in 4 bit? String type ("True", "False")
* `cache_dir`: Where to store the model cache. String type
* `out_dir`: Where to save the outputs. String type
* `hf_token`: Huggingface token for access to the model. String type

<br />

You can run inference on the GPT models with the following command: 
```
python ./model/answer_2_choices_hf.py
```
The file requires the following arguments:
* `model_name`: (Nick)name of the model for savin the results. String type
* `model_name_gpt`: Name of the GPT API model. String type
* `dataset_name`: Name of the dataset in `/data/`. String type
* `use_cot`: Should chain of thought prompting be used? String type ("True" for COT, "False" for Base)
* `answer_type`: "Should we identify correct or incorrect answers? String type ("correct" for DA Strategy, "incorrect" for POE Strategy)
* `out_dir`: Where to save the outputs. String type
* `open_ai_key`: OpenAI API Key
