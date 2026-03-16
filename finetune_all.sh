source .venv/bin/activate

configs=(
  "configs/hermes-3B-lora-entities.yaml"
  "configs/hermes-3B-lora-relations.yaml"
)

for config in "${configs[@]}"; do
  echo "Running finetuning with config: $config"
  python lora_finetune_single_device.py --config "$config"
done