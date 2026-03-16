# GutBrain IE Challenge @ CLEF 2025

`Benedikt Kantz, Peter Walder, Stefan Lengauer, Tobias Schreck`

## Our appraoch

* Use finetuned LLama 3.2 1B or 3B (Hermes as base model seems good?)

## Setup

```sh
# Install dependencies
uv sync --prerelease=allow   
# if you want to use the GPU:
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_BUILD_PARALLEL_LEVEL=8" uv sync
# on a cluster you could start into a interactive environment:
srun --gres=gpu:a40 -c 12 --partition allgroups  --time=10:00  --pty   bash

git submodule update --init --recursive
source .venv/bin/activate
# dowload models (make sure to set you HF token!)
tune download NousResearch/Hermes-3-Llama-3.2-3B  --output-dir models/hermes-3-2-3B
tune download meta-llama/Llama-3.2-3B-Instruct  --output-dir models/llama-3-2-3B-instruct
python manage_models/quantize_all.py

