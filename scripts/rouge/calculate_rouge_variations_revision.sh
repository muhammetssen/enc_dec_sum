#!/bin/bash

HOME_DIR=/Users/bbaykara/PycharmProjects/enc_dec_sum
RESULTS_FOLDER=/Users/bbaykara/PycharmProjects/enc_dec_sum/outputs

for MODEL_NAME in tr_news_berturk_cased_summary tr_news_berturk_uncased_summary tr_news_mbart_summary tr_news_mbert_uncased_summary tr_news_mbert_cased_summary ml_sum_berturk_cased_summary ml_sum_berturk_uncased_summary ml_sum_mbart_summary ml_sum_mbert_uncased_summary ml_sum_mbert_cased_summary combined_tr_berturk_cased_summary combined_tr_berturk_uncased_summary combined_tr_mbart_summary combined_tr_mbert_uncased_summary combined_tr_mbert_cased_summary
do
echo $RESULTS_FOLDER/$MODEL_NAME
$HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --text_outputs_exist True \
      --text_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/text_outputs.csv \
      --rouge_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/rouge_outputs_stem_punc.json \
      --novelty_outputs_file_path $RESULTS_FOLDER/$MODEL_NAME/novelty_outputs_stem_punc.json \
      --use_stemmer_in_rouge True
done

# find ./outputs/test -iname "*_no_punc.json" ! -path "*mt5*" -exec rename -nv 's/no_punc/XXX/g' {} \;