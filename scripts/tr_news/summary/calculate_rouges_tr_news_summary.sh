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
DATASET_NAME=tr_news
TASK_NAME=summary

#for MODEL_NAME in tr_news_mt5_title tr_news_mbart_title tr_news_berturk32k_title tr_news_berturk32k_cased_title tr_news_mbert_uncased_title tr_news_mbert_cased_title
for MODEL_NAME in tr_news_mt5_title tr_news_mbart_title tr_news_berturk32k_title tr_news_berturk32k_cased_title tr_news_mbert_uncased_title tr_news_mbert_cased_title
do
  mkdir -p $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME
  echo $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME

  if [[ "$MODEL_NAME" == *"mt5"* ]]; then
    echo "mt5"
    $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
    --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
    --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
    --text_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/text_outputs.csv \
    --rouge_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/rouge_outputs.json \
    --novelty_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/novelty_outputs.json \
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
    --batch_size 16 \
    --use_stemmer_in_rouge False
  fi
  if [[ "$MODEL_NAME" == *"mbart"* ]]; then
    echo "mbart"
    $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
    --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
    --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
    --text_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/text_outputs.csv \
    --rouge_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/rouge_outputs.json \
    --novelty_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/novely_outputs.json \
    --do_tr_lowercase True \
    --source_column_name content \
    --target_column_name abstract \
    --source_prefix "" \
    --num_beams 4 \
    --ngram_blocking_size 3 \
    --early_stopping True \
    --use_cuda False \
    --max_source_length 768 \
    --max_target_length 128 \
    --batch_size 16 \
    --use_stemmer_in_rouge False
  fi
  if [[ "$MODEL_NAME" == *"bert"* ]]; then
    echo "bert"
    $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
    --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
    --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
    --text_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/text_outputs.csv \
    --rouge_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/rouge_outputs.json \
    --novelty_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/novely_outputs.json \
    --do_tr_lowercase True \
    --source_column_name content \
    --target_column_name abstract \
    --source_prefix "" \
    --num_beams 4 \
    --ngram_blocking_size 3 \
    --early_stopping True \
    --use_cuda False \
    --max_source_length 512 \
    --max_target_length 128 \
    --batch_size 16 \
    --use_stemmer_in_rouge False
  fi
done


