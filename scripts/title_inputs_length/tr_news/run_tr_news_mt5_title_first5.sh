#!/bin/bash
#SBATCH -p long
#SBATCH -J train_t
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=7-00:00:00

module load cuda-10.2.89-gcc-10.2.0-dgnsc3t

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

RUN_NAME=tr_news_mt5_title_first5
HOME_DIR=/cta/users/bbaykara/code/enc_dec_sum
OUTPUTS_DIR=$HOME_DIR/outputs/$RUN_NAME

$HOME_DIR/venv/bin/python  $HOME_DIR/run_summarization.py \
--model_name_or_path google/mt5-base \
--do_train \
--do_eval \
--early_stopping_patience 2 \
--do_predict \
--load_best_model_at_end \
--num_beams 4 \
--max_source_length 256 \
--max_target_length 64 \
--save_strategy epoch \
--evaluation_strategy epoch \
--train_file $HOME_DIR/data/tr_news_raw/train.csv \
--validation_file $HOME_DIR/data/tr_news_raw/validation.csv \
--test_file $HOME_DIR/data/tr_news_raw/test.csv \
--source_prefix "summarize: " \
--output_dir $OUTPUTS_DIR \
--logging_dir $OUTPUTS_DIR/logs \
--overwrite_output_dir \
--predict_with_generate \
--text_column first5 \
--summary_column title \
--do_tr_lowercase \
--preprocessing_num_workers 10 \
--dataloader_num_workers 2 \
--gradient_accumulation_steps 4 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs 10 \
--logging_steps 500 \
--learning_rate 1e-3 \
--warmup_steps 1000 \
--adafactor
