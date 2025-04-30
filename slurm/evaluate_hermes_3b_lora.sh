#!/bin/bash
#SBATCH --job-name=clef_hermes_3b_lora
#SBATCH -c 1
#SBATCH --mem 5500M
#SBATCH -a 0-4%2
#SBATCH --account=bkantz
#SBATCH --output=logs/inference_%A_%a.out
#SBATCH --error=logs/inference_%A_%a.err


export LIBRARY_PATH="/usr/local/cuda-12.6/lib64/stubs/:$LIBRARY_PATH"

# either use --add-rag or --reorder bases on $SLURM_ARRAY_TASK_ID
FLAGS=""
out_file="hermes-3b-lora"
if [ $(($SLURM_ARRAY_TASK_ID%2)) -eq 0 ]; then
    FLAGS="$FLAGS --add-rag"
    echo "Using --add-rag"
    out_file="$out_file-rag"
fi

if [ $(($SLURM_ARRAY_TASK_ID/2)) -eq 0 ]; then
    FLAGS="$FLAGS --reorder"
    echo "Using --reorder" 
    out_file="$out_file-reorder" 
fi
out_file="$out_file.json"

echo "Running with $FLAGS to $out_file"



cd ..
. .venv/bin/activate

python inference.py --model-provider llama --model-spec quants/hermes-3-2-3B-lora.gguf --out-file $out_file $FLAGS 