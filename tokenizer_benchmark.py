import re

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# TR-News dataset variables
data_files = dict()
data_files["train"] = "data/tr_news_raw/train.csv"
data_files["validation"] = "data/tr_news_raw/validation.csv"
data_files["test"] = "data/tr_news_raw/test.csv"

tr_news_dataset = load_dataset("csv", data_files=data_files)
ml_sum_dataset = load_dataset("mlsum", "tu")

# Model names for tokenizers
model_names = ["google/mt5-base", "facebook/mbart-large-cc25", "dbmdz/bert-base-turkish-uncased",
               "dbmdz/bert-base-turkish-128k-uncased", "bert-base-multilingual-uncased"]
print(type(ml_sum_dataset['train']))


def get_num_tokens(tokenizer, text):
    return len(tokenizer.tokenize(text))


def lower_tr(text):
    text = re.sub(r'İ', 'i', text)
    return re.sub(r'I', 'ı', text).lower().strip()


for dataset, dataset_name, column_name in zip([tr_news_dataset, ml_sum_dataset], ['tr_news', 'ml_sum'],
                                              ['title', 'title']):
    for model_name in model_names:
        text_num_tokens = []
        count = 0
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=True, revision="main",
                                                  use_auth_token=None)
        for text in dataset['train'][column_name]:
            text = lower_tr(text)
            text_num_tokens.append(get_num_tokens(tokenizer, text))
            count += 1
        print(
            "Dataset: {} \t Column: {} \t Model: {} \t Avg. num tokens: {} \t Min num token: {} \t Max num token: {}".format(
                dataset_name, column_name,
                model_name,
                np.array(text_num_tokens).mean(),
                np.array(text_num_tokens).min(),
                np.array(text_num_tokens).max()))
