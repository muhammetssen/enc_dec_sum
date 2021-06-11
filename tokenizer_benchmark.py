import re

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# TR-News dataset variables
tr_news_data_files = dict()
tr_news_data_files["train"] = "data/tr_news_raw/train.csv"
tr_news_data_files["validation"] = "data/tr_news_raw/validation.csv"
tr_news_data_files["test"] = "data/tr_news_raw/test.csv"
# MLSum dataset variables
ml_sum_data_files = dict()
ml_sum_data_files["train"] = "data/ml_sum_tr_raw/train.csv"
ml_sum_data_files["validation"] = "data/ml_sum_tr_raw/validation.csv"
ml_sum_data_files["test"] = "data/ml_sum_tr_raw/test.csv"
# Combined-TR dataset variables
combined_tr_data_files = dict()
combined_tr_data_files["train"] = "data/combined_tr_raw/train.csv"
combined_tr_data_files["validation"] = "data/combined_tr_raw/validation.csv"
combined_tr_data_files["test"] = "data/combined_tr_raw/test.csv"

tr_news_dataset = load_dataset("csv", data_files=tr_news_data_files)
ml_sum_dataset = load_dataset("csv", data_files=ml_sum_data_files)
combined_tr_dataset = load_dataset("csv", data_files=combined_tr_data_files)

# Model names for tokenizers
model_names = ["google/mt5-base", "facebook/mbart-large-cc25", "dbmdz/bert-base-turkish-uncased",
               "dbmdz/bert-base-turkish-cased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased"]


def get_num_tokens(tokenizer, text):
    return len(tokenizer.tokenize(text))


def lower_tr(text):
    text = re.sub(r'İ', 'i', text)
    return re.sub(r'I', 'ı', text).lower().strip()


for dataset, dataset_name, column_name in zip([combined_tr_dataset], ['combined_tr'],
                                              ['text']):
    for model_name in model_names:
        text_num_tokens = []
        count = 0
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=True, revision="main",
                                                  use_auth_token=None, strip_accents=False, lowercase=False)
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


for dataset, dataset_name, column_name in zip([tr_news_dataset], ['tr_news'],
                                              ['abstract']):
    for model_name in model_names:
        text_num_tokens = []
        count = 0
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=True, revision="main",
                                                  use_auth_token=None, strip_accents=False, lowercase=False)
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

for dataset, dataset_name, column_name in zip([ml_sum_dataset, combined_tr_dataset], ['ml_sum', 'combined_tr'],
                                              ['summary', 'summary']):
    for model_name in model_names:
        text_num_tokens = []
        count = 0
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=True, revision="main",
                                                  use_auth_token=None, strip_accents=False, lowercase=False)
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

for dataset, dataset_name, column_name in zip([tr_news_dataset, ml_sum_dataset, combined_tr_dataset],
                                              ['tr_news', 'ml_sum', 'combined_tr'],
                                              ['title', 'title', 'title']):
    for model_name in model_names:
        text_num_tokens = []
        count = 0
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=True, revision="main",
                                                  use_auth_token=None, strip_accents=False, lowercase=False)
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

for dataset, dataset_name, column_name in zip([tr_news_dataset, ml_sum_dataset, combined_tr_dataset],
                                              ['tr_news', 'ml_sum', 'combined_tr'],
                                              ['first3', 'first3', 'first3']):
    for model_name in model_names:
        text_num_tokens = []
        count = 0
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=True, revision="main",
                                                  use_auth_token=None, strip_accents=False, lowercase=False)
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
