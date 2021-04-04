import json
import re

import datasets
import pandas as pd
import numpy as np
from nltk import ngrams
from transformers import AutoTokenizer, AutoConfig, EncoderDecoderModel, AutoModelForSeq2SeqLM


class PostMetrics:
    def __init__(self, model_name_or_path, num_workers, dataset_name=None, dataset_version=None,
                 dataset_test_csv_file_path=None, do_tr_lowercase=True, source_column_name="content",
                 target_column_name="abstract", max_source_len=768, max_generation_len=120, beam_size=4,
                 ngram_blocking_size=3, use_cuda=False):
        self.model_name_or_path = model_name_or_path
        self.do_tr_lowercase = do_tr_lowercase
        self.num_workers = num_workers
        self.source_column_name = source_column_name
        self.target_column_name = target_column_name
        self.max_source_len = max_source_len
        self.max_generation_len = max_generation_len
        self.beam_size = beam_size
        self.ngram_blocking_size = ngram_blocking_size
        self.use_cuda = use_cuda
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.rouge = datasets.load_metric("rouge")

        assert dataset_name is not None or dataset_test_csv_file_path is not None, "Either dataset name or file path should be given."
        if dataset_name is not None:
            self.test_data = datasets.load_dataset(dataset_name, dataset_version, split="test")
        else:
            data_files = dict()
            data_files["test"] = dataset_test_csv_file_path
            self.test_data = datasets.load_dataset("csv", data_files=data_files)

        self.test_data = self.test_data.select(range(16))

        self.test_data = self.test_data.map(
            self.preprocess_function,
            batched=True,
            # num_proc=self.num_workers,
        )

    @staticmethod
    def lower_tr(text):
        text = re.sub(r'İ', 'i', text)
        return re.sub(r'I', 'ı', text).lower().strip()

    def preprocess_function(self, examples):
        inputs = examples[self.source_column_name]
        targets = examples[self.target_column_name]

        if self.do_tr_lowercase:
            examples[self.source_column_name] = [self.lower_tr(inp) for inp in inputs]
            examples[self.target_column_name] = [self.lower_tr(trg) for trg in targets]

        return examples

    def load_model_and_tokenizer(self):
        model_name_or_path = self.model_name_or_path
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if "bert" in model_name_or_path:
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.eos_token = tokenizer.sep_token

        if "bert" in model_name_or_path:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name_or_path, model_name_or_path)
            # set special tokens
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

            # sensible parameters for beam search
            model.config.vocab_size = model.config.decoder.vocab_size
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

        if "mbart" in model_name_or_path:
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def generate_beam_summary(self, batch):
        inputs = self.tokenizer(batch[self.source_column_name], padding="max_length", truncation=True,
                                max_length=self.max_source_len,
                                return_tensors="pt")

        if self.use_cuda:
            inputs.input_ids = inputs.input_ids.to("cuda")
            inputs.attention_mask = inputs.attention_mask.to("cuda")

        beam_output = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.max_generation_len,
            num_beams=self.beam_size,
            no_repeat_ngram_size=self.ngram_blocking_size,
            early_stopping=True
        )

        batch["predictions"] = self.tokenizer.batch_decode(beam_output, skip_special_tokens=True)

        return batch

    def calculate_rouge(self, references, predictions):
        rouge_output = self.rouge.compute(predictions=predictions, references=references)
        return rouge_output

    @staticmethod
    def calculate_novelty_ngram_ratio(sources, references, predictions, ngram_size):
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

    def calculate_metrics(self, batch_size, write_results=True, text_outputs_file_path="text_outputs.csv",
                          rouge_outputs_file_path="rouge_output.json", novely_outputs_file_path="novely_outputs.json"):
        results = self.test_data.map(self.generate_beam_summary, batched=True, batch_size=batch_size)
        rouge_output = self.calculate_rouge(results[self.target_column_name], results["predictions"])
        novelty_ratios = self.calculate_novelty_ngram_ratios(results[self.source_column_name],
                                                             results[self.target_column_name], results["predictions"])

        if write_results:
            df = pd.DataFrame({"source": results[self.source_column_name], "target": results[self.target_column_name],
                               "predictions": results["predictions"]})
            df.to_csv(text_outputs_file_path)

            with open(rouge_outputs_file_path, 'w') as fp:
                json.dump(rouge_output, fp)

            with open(novely_outputs_file_path, 'w') as fp:
                json.dump(novelty_ratios, fp)


if __name__ == "__main__":
    post_metrics = PostMetrics("checkpoint_example", num_workers=4, dataset_name="mlsum", dataset_version="tu",
                               source_column_name="text", target_column_name="summary")
    post_metrics.calculate_metrics(batch_size=2)
