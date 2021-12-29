#!/bin/bash

HOME_DIR=/Users/bbaykara/PycharmProjects/enc_dec_sum
RESULTS_FOLDER=/Users/bbaykara/PycharmProjects/enc_dec_sum/outputs

for MODEL_NAME in tr_news_mt5_summary ml_sum_mt5_summary combined_tr_mt5_summary tr_news_mt5_title ml_sum_mt5_title combined_tr_mt5_title
do
$HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --text_outputs_exist True \
      --text_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/text_outputs.csv \
      --rouge_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/rouge_outputs_stem_punc.json \
      --novelty_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/novelty_outputs_stem_punc.json \
      --use_stemmer_in_rouge True
done

for MODEL_NAME in tr_news_mt5_summary ml_sum_mt5_summary combined_tr_mt5_summary tr_news_mt5_title ml_sum_mt5_title combined_tr_mt5_title
do
$HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --text_outputs_exist True \
      --text_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/text_outputs.csv \
      --rouge_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/rouge_outputs_no_stem_punc.json \
      --novelty_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/novelty_outputs_no_stem_punc.json \
      --use_stemmer_in_rouge False
done
