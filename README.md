# GutBrain IE Challenge @ CLEF 2025

`Benedikt Kantz, Peter Walder, Stefan Lengauer, Tobias Schreck`

## Our appraoch

* Use finetuned LLama 3.2 1B or 3B (Hermes as base model seems good?)

## Setup

```sh
# Install dependencies
uv sync --prerelease=allow   
git submodule update --init --recursive
# dowload models (make sure to set you HF token!)
tune download NousResearch/Hermes-3-Llama-3.2-3B  --output-dir models/hermes-3-2-3B
tune download meta-llama/Llama-3.2-3B-Instruct  --output-dir models/llama-3-2-3B-instruct
python quantize_all.py

