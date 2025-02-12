import os
import os.path as osp
import sys
from convert_hf_to_gguf import Model, split_str_to_n_bytes
import argparse
import torch
import gguf
from pathlib import Path

args_parse = argparse.ArgumentParser()
args_parse.add_argument("--model_path", type=str, default="models/")
args_parse.add_argument("--outdir", type=str, default="quants/")
args_parse.add_argument("--outtype", type=str, default="q8_0")

args = args_parse.parse_args()


def quantize_all():
    models = os.listdir(args.model_path)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }
    print("Models to quantize: ", models)
    models= [model for model in models if Path(model).is_dir()]
    for model in models:
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


if __name__ == "__main__":
    quantize_all()
