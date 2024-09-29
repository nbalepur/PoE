# Process of Elimination (PoE)

This repository is the official implementation of "It's Not Easy Being Wrong: Evaluating Process of Elimination Reasoning in Large Language Models", which was accepted to ACL 2024 (findings) and can be viewed [here](https://arxiv.org/abs/2311.07532) on Arxiv.

<p align="center">
  <img src="/images/POE_Intro.png"></img>
</p>

## Overview

This repository contains the code and dataset to run the direct answer and process of elimination strategies, with and without chain of thought, on our four tested commonsense reasoning and scientific reasoning multiple-choice QA datasets.

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
* `model_name`: (Nick)name of the model for saving the results. String type
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
python ./model/answer_2_choices_gpt.py
```
The file requires the following arguments:
* `model_name`: (Nick)name of the model for saving the results. String type
* `model_name_gpt`: Name of the GPT API model. String type
* `dataset_name`: Name of the dataset in `/data/`. String type
* `use_cot`: Should chain of thought prompting be used? String type ("True" for COT, "False" for Base)
* `answer_type`: "Should we identify correct or incorrect answers? String type ("correct" for DA Strategy, "incorrect" for POE Strategy)
* `out_dir`: Where to save the outputs. String type
* `open_ai_key`: OpenAI API Key. String type


## Citation

If you found our code/papers useful, you can cite us as follows:

```{bibtex}
@inproceedings{balepur-etal-2024-easy,
    title = "It{'}s Not Easy Being Wrong: Large Language Models Struggle with Process of Elimination Reasoning",
    author = "Balepur, Nishant  and
      Palta, Shramay  and
      Rudinger, Rachel",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.604",
    doi = "10.18653/v1/2024.findings-acl.604",
    pages = "10143--10166",
    abstract = "Chain-of-thought (COT) prompting can help large language models (LLMs) reason toward correct answers, but its efficacy in reasoning toward incorrect answers is unexplored. This process of elimination (PoE), when used with COT, can enhance self-consistency, interpretability, and tasks such as medical diagnoses of exclusion. Thus, we propose PoE with COT, where LLMs must reason toward incorrect options on multiple-choice questions. We evaluate the ability of GPT-3.5, LLaMA-2, and Falcon to perform PoE with COT on a total of four commonsense and scientific reasoning datasets. We find that the strategy of PoE always underperforms the strategy of choosing the correct answer. The agreement of these strategies is also lower than the self-consistency of each strategy. To study these issues further, we conduct error analyses and give suggestions for future work.",
}
```
