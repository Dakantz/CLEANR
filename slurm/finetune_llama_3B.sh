#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH -c 1
#SBATCH --mem 16G
#SBATCH -a 0-0%1
#SBATCH --account=bkantz
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err


export LIBRARY_PATH="/usr/local/cuda-12.6/lib64/stubs/:$LIBRARY_PATH"


cd ..
source .venv/bin/activate

configs=("configs/llama-3B-lora.yaml")

cfg_id=$(($SLURM_ARRAY_TASK_ID % 2))
selected_config=${configs[$cfg_id]}
echo "config: $selected_config"

tune run lora_finetune_single_device --config $selected_config

