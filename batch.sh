#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH -t 24:00:00

module load 2019

module load Python/2.7.15-intel-2018b

pip install --user -r requirements.txt

module unload 2019

module load pre2019

module unload binutils

module unload GCCcore

module load cuDNN/7.3.1-CUDA-9.0.176

python main.py --do_preprocess

python main.py --do_train --do_evaluate --lr 0.001 --batch_size 20 --nepochs 75
