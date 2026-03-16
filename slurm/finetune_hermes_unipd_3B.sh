#!/bin/bash
#SBATCH --job-name=finetune_clef_hermes
#SBATCH --array=0-1%1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH --gres=gpu:a40 
#SBATCH -p allgroups
#SBATCH --output=logs/finetune_%A_%a.out
#SBATCH --error=logs/finetune_%A_%a.err
#SBATCH --time=24:00:00



cd ..
source .venv/bin/activate

configs=("configs/hermes-3B-lora-entities.yaml" "configs/hermes-3B-lora-relations.yaml")

cfg_id=$(($SLURM_ARRAY_TASK_ID%2))
selected_config=${configs[$cfg_id]}
echo "config: $selected_config"

tune run lora_finetune_single_device --config $selected_config

