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
HOME_DIR=/cta/users/bbaykara/code/enc_dec_sum

$HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
--model_name_or_path $HOME_DIR/outputs/tr_news_mt5_title \
--dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
--text_outputs_file_path $HOME_DIR/rouge_test_mt5_title/text_outputs_gpu.csv \
--rouge_outputs_file_path $HOME_DIR/rouge_test_mt5_title/rouge_outputs_gpu.json \
--novelty_outputs_file_path $HOME_DIR/rouge_test_mt5_title/novely_outputs_gpu.json \
--do_tr_lowercase True \
--source_column_name abstract \
--target_column_name title \
--source_prefix "summarize: " \
--num_beams 4 \
--ngram_blocking_size 3 \
--early_stopping True \
--use_cuda False \
--max_source_length 768 \
--max_target_length 128 \
--batch_size 2 \
--use_stemmer_in_rouge=False
