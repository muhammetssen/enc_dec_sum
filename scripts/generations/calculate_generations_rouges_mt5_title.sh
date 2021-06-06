#!/bin/bash
#SBATCH -p mid
#SBATCH -J inf
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=1-00:00:00

module load cuda-10.2.89-gcc-10.2.0-dgnsc3t

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"
echo "CUDA DEVICES $CUDA_VISIBLE_DEVICES"

HOME_DIR=/cta/users/bbaykara/code/enc_dec_sum
RESULTS_FOLDER=results
MODELS_FOLDER=outputs
TASK_NAME=title

for MODEL_NAME in tr_news_mt5_title/checkpoint-52044 ml_sum_mt5_title/checkpoint-38950 combined_tr_mt5_title/checkpoint-98784
do
  for options in "BEAM_SIZE=1 EARLY_STOPPING=False" "BEAM_SIZE=2 EARLY_STOPPING=False" "BEAM_SIZE=3 EARLY_STOPPING=False" "BEAM_SIZE=4 EARLY_STOPPING=False" "BEAM_SIZE=4 EARLY_STOPPING=True"
  do
    eval $options
    OUTPUTS_PATH=$HOME_DIR/$RESULTS_FOLDER/generations/$TASK_NAME/$MODEL_NAME/beam_size_$BEAM_SIZE/early_stopping_$EARLY_STOPPING
    mkdir -p $OUTPUTS_PATH
    echo $OUTPUTS_PATH

    if [[ "$MODEL_NAME" == *"tr_news"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name abstract \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams $BEAM_SIZE \
      --ngram_blocking_size 3 \
      --early_stopping $EARLY_STOPPING \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
    if [[ "$MODEL_NAME" == *"ml_sum"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/ml_sum_tr_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name summary \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams $BEAM_SIZE \
      --ngram_blocking_size 3 \
      --early_stopping $EARLY_STOPPING \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
    if [[ "$MODEL_NAME" == *"combined_tr"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/combined_tr_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name summary \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams $BEAM_SIZE \
      --ngram_blocking_size 3 \
      --early_stopping $EARLY_STOPPING \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
  done
done

TASK_NAME=title_first3

for MODEL_NAME in tr_news_mt5_title_first3/checkpoint-52044 ml_sum_mt5_title_first3/checkpoint-38950 combined_tr_mt5_title_first3/checkpoint-98784
do
  for options in "BEAM_SIZE=1 EARLY_STOPPING=False" "BEAM_SIZE=2 EARLY_STOPPING=False" "BEAM_SIZE=3 EARLY_STOPPING=False" "BEAM_SIZE=4 EARLY_STOPPING=False" "BEAM_SIZE=4 EARLY_STOPPING=True"
  do
    eval $options
    OUTPUTS_PATH=$HOME_DIR/$RESULTS_FOLDER/generations/$TASK_NAME/$MODEL_NAME/beam_size_$BEAM_SIZE/early_stopping_$EARLY_STOPPING
    mkdir -p $OUTPUTS_PATH
    echo $OUTPUTS_PATH

    if [[ "$MODEL_NAME" == *"tr_news"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/tr_news_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name first3 \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams $BEAM_SIZE \
      --ngram_blocking_size 3 \
      --early_stopping $EARLY_STOPPING \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
    if [[ "$MODEL_NAME" == *"ml_sum"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/ml_sum_tr_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name first3 \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams $BEAM_SIZE \
      --ngram_blocking_size 3 \
      --early_stopping $EARLY_STOPPING \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
    if [[ "$MODEL_NAME" == *"combined_tr"* ]]; then
      $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --model_name_or_path $HOME_DIR/$MODELS_FOLDER/$MODEL_NAME \
      --dataset_test_csv_file_path $HOME_DIR/data/combined_tr_raw/test.csv \
      --text_outputs_file_path $OUTPUTS_PATH/text_outputs.csv \
      --rouge_outputs_file_path $OUTPUTS_PATH/rouge_outputs.json \
      --novelty_outputs_file_path $OUTPUTS_PATH/novely_outputs.json \
      --do_tr_lowercase True \
      --source_column_name first3 \
      --target_column_name title \
      --source_prefix "summarize: " \
      --num_beams $BEAM_SIZE \
      --ngram_blocking_size 3 \
      --early_stopping $EARLY_STOPPING \
      --use_cuda True \
      --max_source_length 256 \
      --max_target_length 64 \
      --batch_size 16 \
      --use_stemmer_in_rouge False
    fi
  done
done
