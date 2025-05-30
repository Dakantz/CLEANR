#!/bin/bash
#SBATCH --job-name=clef_hermes_3b_NER
#SBATCH -c 1
#SBATCH --mem 5500M
#SBATCH -a 0-16%2
#SBATCH --account=bkantz
#SBATCH --output=logs/inference_%A_%a.out
#SBATCH --error=logs/inference_%A_%a.err


export LIBRARY_PATH="/usr/local/cuda-12.6/lib64/stubs/:$LIBRARY_PATH"

# either use --add-rag or --reorder bases on $SLURM_ARRAY_TASK_ID
FLAGS=""
out_file="hermes-3b"
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

if [ $(($SLURM_ARRAY_TASK_ID/4)) -eq 0 ]; then
    FLAGS="$FLAGS --entity-labels"
    echo "Using --entity-labels" 
    out_file="$out_file-entity-labels" 
fi

if [ $(($SLURM_ARRAY_TASK_ID/8)) -eq 0 ]; then
    FLAGS="$FLAGS --gen-tokens=512"
    echo "Using --gen-tokens=512" 
    out_file="$out_file-low-tokens" 
fi

out_file="$out_file.json"

echo "Running with $FLAGS to $out_file"



cd ..
. .venv/bin/activate

python inference_ner.py --model-provider llama --model-spec NousResearch/Hermes-3-Llama-3.2-3B-GGUF --out-file $out_file $FLAGS 