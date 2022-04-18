import argparse
import json
import os
import re
import sys

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, EncoderDecoderModel, AutoModelForSeq2SeqLM, MBartTokenizerFast, \
    set_seed

set_seed(42)


def lower_tr(text):
    text = re.sub(r'İ', 'i', text)
    return re.sub(r'I', 'ı', text).lower().strip()


class ApiHelper:
    def __init__(self, model_name_or_path, do_tr_lowercase=True, source_prefix=None, max_source_length=768,
                 max_target_length=120, num_beams=4, ngram_blocking_size=3, early_stopping=None, use_cuda=False,
                 batch_size=2, language="tr",source_column_name = "input"):
        self.model_name_or_path = model_name_or_path
        self.do_tr_lowercase = do_tr_lowercase
        self.source_prefix = source_prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.ngram_blocking_size = ngram_blocking_size
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.language = language
        self.source_column_name = source_column_name

        self.model, self.tokenizer = self.load_model_and_tokenizer()

        if self.use_cuda:
            self.model = self.model.to("cuda")

    def preprocess_function(self, examples):
        if self.do_tr_lowercase:
            examples[self.source_column_name] = [lower_tr(inp) for inp in examples[self.source_column_name]]

        return examples

    def load_model_and_tokenizer(self):
        model_name_or_path = self.model_name_or_path
        config = AutoConfig.from_pretrained(
            model_name_or_path
        )
        if "mbart" in model_name_or_path:
            tokenizer = MBartTokenizerFast.from_pretrained(
                model_name_or_path,
                src_lang="tr_TR",
                tgt_lang="tr_TR")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=True,
                strip_accents=False,
                lowercase=False
            )

        if "bert" in model_name_or_path:
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.eos_token = tokenizer.sep_token

        if "bert" in model_name_or_path:
            model = EncoderDecoderModel.from_pretrained(model_name_or_path)
            # set special tokens
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

            # sensible parameters for beam search
            model.config.vocab_size = model.config.decoder.vocab_size
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                config=config,
            )

        if "mbart" in model_name_or_path:
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        return model, tokenizer

    def generate_summary(self, batch):
        tokenizer = self.tokenizer
        inputs = batch[self.source_column_name]
        inputs = [self.source_prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=self.max_source_length, padding="max_length", truncation=True,
                                 return_tensors="pt")

        if self.use_cuda:
            model_inputs.input_ids = model_inputs.input_ids.to("cuda")
            model_inputs.attention_mask = model_inputs.attention_mask.to("cuda")

        output = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_length=self.max_target_length,
            num_beams=self.num_beams,
            no_repeat_ngram_size=self.ngram_blocking_size,
            early_stopping=self.early_stopping
        )

        batch["predictions"] = self.tokenizer.batch_decode(output, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True)

        return batch

    def predict(self):
        results = self.test_data.map(self.generate_summary, batched=True, batch_size=self.batch_size,
                                     load_from_cache_file=False)
