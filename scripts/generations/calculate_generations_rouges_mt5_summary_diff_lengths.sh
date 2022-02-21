#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A bbaykara
#SBATCH -J inf
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL

module load centos7.3/lib/cuda/10.1

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

HOME_DIR=/truba/home/bbaykara/code/enc_dec_sum
RESULTS_FOLDER=results
MODELS_FOLDER=outputs
TASK_NAME=summary

for MODEL_NAME in tr_news_mt5_summary/checkpoint-60718 ml_sum_mt5_summary/checkpoint-62320 combined_tr_mt5_summary/checkpoint-131712
do
  for LENGTH in 64 96 128 160 192 224
  do
    OUTPUTS_PATH=$HOME_DIR/$RESULTS_FOLDER/generations/$TASK_NAME/$MODEL_NAME/length_$LENGTH
    mkdir -p $OUTPUTS_PATH
    echo $OUTPUTS_PATH

    if [[ "$MODEL_NAME" == *"tr_news"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novelty_outputs.json \
      --do_tr_lowercase True \
      --source_column_name content \
      --target_column_name abstract \
      --source_prefix "summarize: " \
      --num_beams 4 \
      --ngram_blocking_size 3 \
      --early_stopping False \
      --use_cuda True \
      --max_source_length 768 \
      --max_target_length $LENGTH \
      --batch_size 16 \
      --use_stemmer_in_rouge True
    fi
    if [[ "$MODEL_NAME" == *"ml_sum"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/ml_sum_tr_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novelty_outputs.json \
      --do_tr_lowercase True \
      --source_column_name text \
      --target_column_name summary \
      --source_prefix "summarize: " \
      --num_beams 4 \
      --ngram_blocking_size 3 \
      --early_stopping False \
      --use_cuda True \
      --max_source_length 768 \
      --max_target_length $LENGTH \
      --batch_size 16 \
      --use_stemmer_in_rouge True
    fi
    if [[ "$MODEL_NAME" == *"combined_tr"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/combined_tr_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novelty_outputs.json \
      --do_tr_lowercase True \
      --source_column_name text \
      --target_column_name summary \
      --source_prefix "summarize: " \
      --num_beams 4 \
      --ngram_blocking_size 3 \
      --early_stopping False \
      --use_cuda True \
      --max_source_length 768 \
      --max_target_length $LENGTH \
      --batch_size 16 \
      --use_stemmer_in_rouge True
    fi
  done
done