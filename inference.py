# %%
from llama_cpp import Llama
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel

from constrerl.annotator import (
    AnnotatedArticle,
    Annotator,
    AnnotatorHelper,
    load_train,
    load_test,
    StringERLModel,
    prepare_for_eval,
)
from constrerl.erl_schema import convert_to_output

# %%
import os
import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-provider", type=str, default="llama")
    parser.add_argument(
        "--model-spec", type=str, default="quants/llama-3-2-1B-instruct-lora.gguf"
    )
    parser.add_argument(
        "--data-path", type=str, default="data/annotations/prepared_dev_train.json"
    )
    parser.add_argument(
        "--eval-path", type=str, default="data/Annotations/Dev/json_format/dev.json"
    )
    parser.add_argument("--out-path", type=str, default="data/results_dev")
    parser.add_argument("--out-file", type=str, default="dev_out.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gen-tokens", type=int, default=512)
    parser.add_argument("--ctx", type=int, default=8196)
    parser.add_argument("--add-rag", default=False, action="store_true")
    parser.add_argument("--reorder", default=False, action="store_true")
    parser.add_argument("--entity-labels", default=False, action="store_true")
    args = parser.parse_args()
    print("Starting with", args)
    llm: BaseChatModel = None
    model: Llama = None
    match args.model_provider:
        case "openai":
            # llm = init_chat_model("ft:gpt-4o-mini-2024-07-18:tu-graz-hereditary:gutbrain-ie-finetune:B5qr9cGV", model_provider="openai")
            llm = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")
        case "llama":
            if args.model_spec.endswith(".gguf"):
                model_path = args.model_spec  # "quants/llama-3-2-1B-instruct-lora.gguf"
                model = Llama(
                    model_path,
                    n_gpu_layers=-1,
                    n_ctx=args.ctx,
                    temperature=0.1,
                    # draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10),
                )
            else:
                model = Llama.from_pretrained(
                    args.model_spec,
                    filename="*.Q8_0.gguf",
                    n_gpu_layers=-1,
                    n_ctx=args.ctx,
                    temperature=0.1,
                )
    # %%

    # %%

    # %%

    # %%
    data_path = args.data_path
    out_path = Path(args.out_path) / args.out_file
    annotator = AnnotatorHelper(
        model=model,
        gen_tokens=args.gen_tokens,
        add_rag=args.add_rag,
        reorder=args.reorder,
        top_k=args.top_k,
        add_entity_labels=args.entity_labels,
    )
    annotator.load_articles_from_path(Path(data_path))

    with open(args.eval_path, "r") as f:
        eval_set = json.load(f)
    eval_set = {
        id: AnnotatedArticle.model_validate(article) for id, article in eval_set.items()
    }

    # %%
    annotations: dict[str, AnnotatedArticle] = annotator.annotate(
        {id: article.metadata for id, article in list(eval_set.items())}
    )
    annotator.add_concept_uris(annotations)

    output_data = prepare_for_eval(annotations)
    # %%
    with open(out_path, "w") as f:
        json.dump(output_data, f)
    # %%
    print("Done")
