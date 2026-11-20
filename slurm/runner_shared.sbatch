#!/bin/bash
#SBATCH -A m5451_g
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=2
#SBATCH -t 06:00:00
#SBATCH -J hc-runner
#SBATCH -o logs/runner_%j.out
cd /pscratch/sd/m/mahf708/hackathon-circuits
bash lib/runner.sh data/queue 2 1
