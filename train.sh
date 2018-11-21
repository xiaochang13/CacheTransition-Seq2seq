#!/bin/bash
#SBATCH -J hard_s --partition=gpu --gres=gpu:1 --time=5-00:00:00 --output=train.out_separate --error=train.err_separate
#SBATCH --mem=35GB
#SBATCH -c 1
#SBATCH --reservation=xpeng3-may2018


module load graphviz

CONFIG_FILE=./config_files/config_"$1".json
#./config_files/config_uniform5.json
python NP2P_trainer.py --config_path ${CONFIG_FILE}

