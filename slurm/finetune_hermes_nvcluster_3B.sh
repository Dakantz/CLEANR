#!/bin/bash
#SBATCH --job-name=finetune_clef_hermes
#SBATCH -c 2
#SBATCH --mem 32G
#SBATCH --gres=gpu
#SBATCH -p ivc
#SBATCH --output=logs/finetune_%A_%a.out
#SBATCH --error=logs/finetune_%A_%a.err


# export LIBRARY_PATH="/usr/local/cuda-12.6/lib64/stubs/:$LIBRARY_PATH"


cd ..
source .venv/bin/activate

configs=("configs/hermes-3B-lora.yaml")

cfg_id=$(($SLURM_ARRAY_TASK_ID%2))
selected_config=${configs[$cfg_id]}
echo "config: $selected_config"

tune run lora_finetune_single_device --config $selected_config

