#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A bbaykara
#SBATCH -J berturk_combined_tr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL

module load centos7.3/lib/cuda/10.1

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

RUN_NAME=combined_tr_berturk32k_summary
OUTPUTS_DIR=/truba/home/bbaykara/code/enc_dec_sum/outputs/$RUN_NAME

/truba/home/bbaykara/code/enc_dec_sum/venv/bin/python -m torch.distributed.launch --nproc_per_node=4 /truba/home/bbaykara/code/enc_dec_sum/run_summarization.py \
--model_name_or_path dbmdz/bert-base-turkish-uncased \
--pad_to_max_length True \
--do_train \
--do_eval \
--early_stopping_patience 2 \
--do_predict \
--load_best_model_at_end \
--num_beams 4 \
--max_source_length 512 \
--max_target_length 128 \
--save_strategy epoch \
--evaluation_strategy epoch \
--train_file /truba/home/bbaykara/code/enc_dec_sum/data/combined_tr_raw/train.csv \
--validation_file /truba/home/bbaykara/code/enc_dec_sum/data/combined_tr_raw/validation.csv \
--test_file /truba/home/bbaykara/code/enc_dec_sum/data/combined_tr_raw/test.csv \
--output_dir $OUTPUTS_DIR \
--logging_dir $OUTPUTS_DIR/logs \
--overwrite_output_dir \
--predict_with_generate \
--text_column text \
--summary_column summary \
--do_tr_lowercase \
--preprocessing_num_workers 20 \
--dataloader_num_workers 2 \
--gradient_accumulation_steps 1 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs 10 \
--logging_steps 500 \
--warmup_steps 1000 \
--sharded_ddp simple 
