#!/bin/bash
#SBATCH -p mid
#SBATCH -J inf
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
##SBATCH --constraint=v100
#SBATCH --time=1-00:00:00

module load cuda-10.2.89-gcc-10.2.0-dgnsc3t

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

HOME_DIR=/cta/users/bbaykara/code/enc_dec_sum
RESULTS_FOLDER=results
MODELS_FOLDER=outputs
DATASET_NAME=title_length_ablation
TASK_NAME=title_length_ablation

for MODEL_NAME in tr_news_mt5_title_first1 tr_news_mt5_title_first2 ml_sum_mt5_title_first1 ml_sum_mt5_title_first2 combined_tr_mt5_title_first1 combined_tr_mt5_title_first2
do
  for MODEL_NAME in $(find $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME -maxdepth 1 -type d)
  do
    MODEL_NAME=${MODEL_NAME#$HOME_DIR/$MODELS_FOLDER/}
    if [[ "$MODEL_NAME" == *"tr_news"* ]]; then
      echo "tr_news_raw"
      DATASET_NAME="tr_news_raw"
    fi
    if [[ "$MODEL_NAME" == *"ml_sum"* ]]; then
      echo "ml_sum_tr_raw"
      DATASET_NAME="ml_sum_tr_raw"
    fi
    if [[ "$MODEL_NAME" == *"combined_tr"* ]]; then
      echo "combined_tr"
      DATASET_NAME="combined_tr_raw"
    fi

    if [[ "$MODEL_NAME" == *"first1"* ]]; then
      echo "first1"
      SOURCE_FIELD="first1"
    fi
    if [[ "$MODEL_NAME" == *"first2"* ]]; then
      echo "first2"
      SOURCE_FIELD="first2"
    fi

    mkdir -p $HOME_DIR/$RESULTS_FOLDER/$TASK_NAME/$MODEL_NAME
    echo $HOME_DIR/$RESULTS_FOLDER/$TASK_NAME/$MODEL_NAME

    $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
    --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
    --dataset_test_csv_file_path $HOME_DIR/data/$DATASET_NAME/test.csv \
    --text_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$TASK_NAME/$MODEL_NAME/text_outputs.csv \
    --rouge_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$TASK_NAME/$MODEL_NAME/rouge_outputs.json \
    --novelty_outputs_file_path $HOME_DIR/$RESULTS_FOLDER/$TASK_NAME/$MODEL_NAME/novely_outputs.json \
    --do_tr_lowercase True \
    --source_column_name $SOURCE_FIELD \
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
  done
done


