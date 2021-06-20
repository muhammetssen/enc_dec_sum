#!/bin/bash
#SBATCH -p mid1
#SBATCH -A bbaykara
#SBATCH -J results
#SBATCH -c 4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL

HOME_DIR=/truba/home/bbaykara/code/enc_dec_sum
RESULTS_FOLDER=$HOME_DIR/results

find $RESULTS_FOLDER -name text_outputs.csv | while read file;
do
  echo $file
  folder="$(dirname "${file}")"
  echo $folder
  $HOME_DIR/venv/bin/python $HOME_DIR/post_metrics.py \
      --text_outputs_exist True \
      --text_outputs_file_path $folder/text_outputs.csv \
      --rouge_outputs_file_path $folder/rouge_outputs.json \
      --novelty_outputs_file_path $folder/novelty_outputs.json \
      --use_stemmer_in_rouge True
done