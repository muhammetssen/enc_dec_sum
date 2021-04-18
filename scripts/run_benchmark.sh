#!/bin/bash
#SBATCH -p long
#SBATCH -J gpu_benchmark
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
/cta/users/bbaykara/code/enc_dec_sum/venv/bin/python /cta/users/bbaykara/code/enc_dec_sum/benchmark.py
