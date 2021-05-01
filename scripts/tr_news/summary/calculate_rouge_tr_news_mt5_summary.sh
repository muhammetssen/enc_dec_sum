#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A bbaykara
#SBATCH -J rouge_mt5_tr_news
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL

#module load centos7.3/lib/cuda/10.1

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

/truba/home/bbaykara/code/enc_dec_sum/venv/bin/python /truba/home/bbaykara/code/enc_dec_sum/post_metrics.py \
--model_name_or_path /truba/home/bbaykara/code/enc_dec_sum/outputs/tr_news_mt5_summary \
--dataset_test_csv_file_path /truba/home/bbaykara/code/enc_dec_sum/data/tr_news_raw/test.csv \
--text_outputs_file_path /truba/home/bbaykara/code/enc_dec_sum/rouge_test_mt5/text_outputs_gpu.csv \
--rouge_outputs_file_path /truba/home/bbaykara/code/enc_dec_sum/rouge_test_mt5/rouge_outputs_gpu.json \
--novelty_outputs_file_path /truba/home/bbaykara/code/enc_dec_sum/rouge_test_mt5/novely_outputs_gpu.json \
--do_tr_lowercase True \
--source_column_name content \
--target_column_name abstract \
--source_prefix "summarize: " \
--num_beams 4 \
--ngram_blocking_size 3 \
--early_stopping True \
--use_cuda False \
--max_source_length 768 \
--max_target_length 128 \
--batch_size 2 \
--use_stemmer_in_rouge=False
