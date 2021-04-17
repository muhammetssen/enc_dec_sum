#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A bbaykara
#SBATCH -J mt5
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
/truba/home/bbaykara/code/enc_dec_sum/venv/bin/python /truba/home/bbaykara/code/enc_dec_sum/benchmark.py
