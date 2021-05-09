#!/bin/bash
#SBATCH -p long
#SBATCH -J inf
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
##SBATCH --constraint=v100
#SBATCH --time=7-00:00:00

module load cuda-10.2.89-gcc-10.2.0-dgnsc3t

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

HOME_DIR=/cta/users/bbaykara/code/enc_dec_sum
RESULTS_FOLDER=results
MODELS_FOLDER=outputs
DATASET_NAME=tr_news
TASK_NAME=title

#for MODEL_NAME in tr_news_mt5_title tr_news_mbart_title tr_news_berturk32k_title tr_news_berturk32k_cased_title tr_news_mbert_uncased_title tr_news_mbert_cased_title
for MODEL_NAME in tr_news_mt5_title tr_news_berturk32k_title tr_news_berturk32k_cased_title tr_news_mbert_uncased_title tr_news_mbert_cased_title
do
  for MODEL_NAME in $(find $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME -maxdepth 1 -type d)
  do
    MODEL_NAME=${MODEL_NAME#$HOME_DIR/$MODELS_FOLDER/}
    mkdir -p $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME
    echo $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME

    if [[ "$MODEL_NAME" == *"mt5"* ]]; then
      echo "mt5"
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
      --text_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/text_outputs.csv \
      --rouge_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/rouge_outputs.json \
      --novelty_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$DATASET_NAME/$TASK_NAME/$MODEL_NAME/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name abstract \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams 4 \
      --ngram_blocking_size 3 \
      --early_stopping False \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
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
      --source_column_name abstract \
      --target_column_name title \
      --source_prefix "" \
      --num_beams 4 \
      --ngram_blocking_size 3 \
      --early_stopping False \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
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
      --source_column_name abstract \
      --target_column_name title \
      --source_prefix "" \
      --num_beams 4 \
      --ngram_blocking_size 3 \
      --early_stopping False \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
  done
done


