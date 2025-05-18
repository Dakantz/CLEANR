import os
import os.path as osp
import sys
from convert_hf_to_gguf import Model, split_str_to_n_bytes
import argparse
import torch
import gguf
from pathlib import Path
from torch import Tensor
import tqdm

args_parse = argparse.ArgumentParser()
args_parse.add_argument("--model_path", type=str, default="models/")
args_parse.add_argument("--finetunes_path", type=str, default="../finetunes/")
args_parse.add_argument("--outdir", type=str, default="../quants/")
args_parse.add_argument("--outtype", type=str, default="q8_0")

args = args_parse.parse_args()


def quantize_all():
    models = os.listdir(args.model_path)
    if not os.path.exists(args.finetunes_path):
        os.mkdir(args.finetunes_path)
    finetunes = os.listdir(args.finetunes_path)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }
    models = [model for model in models if (Path(args.model_path) / model).is_dir()]
    print("Models to quantize: ", models)
    for model in tqdm.tqdm(models, desc="Quantizing models"):
        print("Quantizing model: ", model)
        dir_model = Path(args.model_path) / model
        hparams = Model.load_hparams(dir_model)
        dir_quant = Path(args.outdir) / (model + ".gguf")
        with torch.inference_mode():
            output_type = ftype_map[args.outtype]
            model_architecture = hparams["architectures"][0]

            try:
                model_class = Model.from_model_architecture(model_architecture)
            except NotImplementedError:
                print(f"Model {model_architecture} is not supported")
                sys.exit(1)
            if not dir_quant.exists():
                model_instance = model_class(
                    dir_model=dir_model,
                    ftype=output_type,
                    fname_out=dir_quant,
                    is_big_endian=False,
                    use_temp_file=False,
                    eager=False,
                    split_max_tensors=0,
                    split_max_size=split_str_to_n_bytes("0"),
                    dry_run=False,
                    small_first_shard=False,
                )

                print("Exporting model...")
                model_instance.write()
                # out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
                print(f"Model successfully exported to {dir_quant}")
            else:
                print(f"Model {model} already quantized, skipping...")
            finetunes_for_model = [
                finetune for finetune in finetunes if model in finetune
            ]
            if len(finetunes_for_model) == 0:
                print("No finetunes found for model: ", model)
                continue
            print("Finetunes found for model: ", finetunes_for_model)
            for finetune in tqdm.tqdm(finetunes_for_model, desc="Quantizing finetunes"):
                print("Quantizing finetune: ", finetune)
                dir_lora_all: Path = Path(args.finetunes_path) / finetune
                last_epoch_dir = sorted(dir_lora_all.glob("epoch_*"))[-1]
                print("Last epoch dir: ", last_epoch_dir)
                dir_lora = last_epoch_dir
                dir_quant = Path(args.outdir) / (finetune + ".gguf")
                # we assume a merged model is in the same directory
                # we have to rename all ft-models to model
                ft_safetensors = dir_lora.glob("ft-model*")
                for ft_safetensor in ft_safetensors:
                    ft_safetensor.rename(
                        ft_safetensor.parent
                        / ft_safetensor.name.replace("ft-model", "model")
                    )

                hparams = Model.load_hparams(dir_lora)

                model_instance = model_class(
                    dir_model=dir_lora,
                    ftype=output_type,
                    fname_out=dir_quant,
                    is_big_endian=False,
                    use_temp_file=False,
                    eager=False,
                    split_max_tensors=0,
                    split_max_size=split_str_to_n_bytes("0"),
                    dry_run=False,
                    small_first_shard=False,
                )
                print("Exporting model...")
                model_instance.write()
                print(f"Model successfully exported to {model_instance.fname_out}")


if __name__ == "__main__":
    quantize_all()
