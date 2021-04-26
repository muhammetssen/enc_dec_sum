import argparse
import json
import re

import datasets
import numpy as np
import pandas as pd
from nltk import ngrams
from transformers import AutoTokenizer, AutoConfig, EncoderDecoderModel, AutoModelForSeq2SeqLM, MBartTokenizerFast


def lower_tr(text):
    text = re.sub(r'İ', 'i', text)
    return re.sub(r'I', 'ı', text).lower().strip()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PostMetrics:
    def __init__(self, model_name_or_path, dataset_name=None, dataset_version=None,
                 dataset_test_csv_file_path=None, do_tr_lowercase=True, source_column_name="content",
                 target_column_name="abstract", source_prefix=None, max_source_length=768, max_target_length=120,
                 beam_size=None,
                 ngram_blocking_size=None, early_stopping=None, use_cuda=False, batch_size=2, write_results=True,
                 text_outputs_file_path="text_outputs.csv",
                 rouge_outputs_file_path="rouge_outputs.json",
                 novelty_outputs_file_path="novelty_outputs.json"):
        self.model_name_or_path = model_name_or_path
        self.do_tr_lowercase = do_tr_lowercase
        self.source_column_name = source_column_name
        self.target_column_name = target_column_name
        self.source_prefix = source_prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.beam_size = beam_size
        self.early_stopping = early_stopping
        self.ngram_blocking_size = ngram_blocking_size
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.write_results = write_results
        self.text_outputs_file_path = text_outputs_file_path
        self.rouge_outputs_file_path = rouge_outputs_file_path
        self.novelty_outputs_file_path = novelty_outputs_file_path

        self.rouge = datasets.load_metric("rouge")
        self.model, self.tokenizer = self.load_model_and_tokenizer()

        if self.use_cuda:
            self.model.to("cuda")

        assert dataset_name is not None or dataset_test_csv_file_path is not None, "Either dataset name or file path should be given."
        if dataset_name is not None:
            self.test_data = datasets.load_dataset(dataset_name, dataset_version, split="test")
        else:
            data_files = dict()
            data_files["test"] = dataset_test_csv_file_path
            self.test_data = datasets.load_dataset("csv", data_files=data_files, split="test")

        # self.test_data = self.test_data.select(range(16))

        columns_to_remove = list(
            set(self.test_data.column_names) - set([self.source_column_name, self.target_column_name]))
        self.test_data = self.test_data.map(
            self.preprocess_function,
            batched=True,
            remove_columns=columns_to_remove,
        )

    def preprocess_function(self, examples):
        if self.do_tr_lowercase:
            examples[self.source_column_name] = [lower_tr(inp) for inp in examples[self.source_column_name]]
            examples[self.target_column_name] = [lower_tr(trg) for trg in examples[self.target_column_name]]

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
            num_beams=self.beam_size,
            no_repeat_ngram_size=self.ngram_blocking_size,
            early_stopping=self.early_stopping
        )

        batch["predictions"] = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return batch

    def calculate_rouge(self, references, predictions):
        rouge_output = self.rouge.compute(predictions=predictions, references=references)
        rouge_output["R1_F_avg"] = rouge_output["rouge1"].mid.fmeasure
        rouge_output["R2_F_avg"] = rouge_output["rouge2"].mid.fmeasure
        rouge_output["RL_F_avg"] = rouge_output["rougeL"].mid.fmeasure
        rouge_output["RLsum_F_avg"] = rouge_output["rougeLsum"].mid.fmeasure

        return rouge_output

    def calculate_novelty_ngram_ratio(self, sources, references, predictions, ngram_size):
        prediction_novelty_ratios = []
        reference_novelty_ratios = []

        for source, reference, prediction in zip(sources, references, predictions):
            prediction_ngrams = set(ngrams(prediction.split(), ngram_size))
            reference_ngrams = set(ngrams(reference.split(), ngram_size))
            source_ngrams = set(ngrams(source.split(), ngram_size))

            joint = prediction_ngrams.intersection(source_ngrams)
            novel = prediction_ngrams - joint
            prediction_novelty_ratios.append(len(novel) / (len(prediction.split()) + 1e-6))

            joint = reference_ngrams.intersection(source_ngrams)
            novel = reference_ngrams - joint
            reference_novelty_ratios.append(len(novel) / (len(reference.split()) + 1e-6))

        return {"prediction_novelty_ratios": np.array(prediction_novelty_ratios).mean(),
                "reference_novelty_ratios": np.array(reference_novelty_ratios).mean()}

    def calculate_novelty_ngram_ratios(self, sources, references, predictions):
        unigram_results = self.calculate_novelty_ngram_ratio(sources, references, predictions, 1)
        bigram_results = self.calculate_novelty_ngram_ratio(sources, references, predictions, 2)
        trigram_results = self.calculate_novelty_ngram_ratio(sources, references, predictions, 3)

        return {"unigram": unigram_results, "bigram": bigram_results, "trigram": trigram_results}

    def calculate_metrics(self):
        results = self.test_data.map(self.generate_summary, batched=True, batch_size=self.batch_size)
        rouge_output = self.calculate_rouge(results[self.target_column_name], results["predictions"])
        novelty_ratios = self.calculate_novelty_ngram_ratios(results[self.source_column_name],
                                                             results[self.target_column_name], results["predictions"])

        if self.write_results:
            df = pd.DataFrame({"source": results[self.source_column_name], "target": results[self.target_column_name],
                               "predictions": results["predictions"]})
            df.to_csv(self.text_outputs_file_path)

            with open(self.rouge_outputs_file_path, 'w') as fp:
                json.dump(rouge_output, fp)

            with open(self.novelty_outputs_file_path, 'w') as fp:
                json.dump(novelty_ratios, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_version", default=None, type=str)
    parser.add_argument("--dataset_test_csv_file_path", default=None, type=str)
    parser.add_argument("--do_tr_lowercase", default=True, type=str2bool)
    parser.add_argument("--source_column_name", default="text", type=str)
    parser.add_argument("--target_column_name", default="summary", type=str)
    parser.add_argument("--source_prefix", default="summary: ", type=str)
    parser.add_argument("--max_source_length", default=768, type=int)
    parser.add_argument("--max_target_length", default=120, type=int)
    parser.add_argument("--beam_size", default=None, type=int)
    parser.add_argument("--ngram_blocking_size", default=None, type=int)
    parser.add_argument("--early_stopping", default=None, type=str2bool)
    parser.add_argument("--use_cuda", default=False, type=str2bool)
    parser.add_argument("--write_results", default=True, type=str2bool)
    parser.add_argument("--text_outputs_file_path", default="text_outputs.csv", type=str)
    parser.add_argument("--rouge_outputs_file_path", default="rouge_outputs.json", type=str)
    parser.add_argument("--novelty_outputs_file_path", default="novelty_outputs.json", type=str)
    parser.add_argument("--batch_size", default=2, type=int)

    args, unknown = parser.parse_known_args()
    print(args)

    post_metrics = PostMetrics(args.model_name_or_path, dataset_name=args.dataset_name,
                               dataset_version=args.dataset_version,
                               dataset_test_csv_file_path=args.dataset_test_csv_file_path,
                               do_tr_lowercase=args.do_tr_lowercase, source_column_name=args.source_column_name,
                               target_column_name=args.target_column_name, source_prefix=args.source_prefix,
                               max_source_length=args.max_source_length,
                               max_target_length=args.max_target_length, beam_size=args.beam_size,
                               ngram_blocking_size=args.ngram_blocking_size, use_cuda=args.use_cuda,
                               batch_size=args.batch_size, write_results=args.write_results,
                               text_outputs_file_path=args.text_outputs_file_path,
                               rouge_outputs_file_path=args.rouge_outputs_file_path,
                               novelty_outputs_file_path=args.novelty_outputs_file_path)

    post_metrics.calculate_metrics()
