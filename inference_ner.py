# %%
from llama_cpp import Llama
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel

from constrerl.annotator import (
    load_train,
    load_test,
)

# %%
import os
import argparse
import json
from pathlib import Path

from constrerl.ner import NERCleanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-provider", type=str, default="openai")
    parser.add_argument(
        "--model-spec", type=str, default="quants/llama-3-2-1B-instruct-lora.gguf"
    )
    parser.add_argument(
        # "--data-path", type=str, default="data/articles/articles_test.json"
        "--data-path",
        type=str,
        default="data/articles/articles_dev.json",
    )
    parser.add_argument("--out-path", type=str, default="data/results_ner_dev")
    parser.add_argument("--out-file", type=str, default="dev_out.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gen-tokens", type=int, default=1024)
    parser.add_argument("--add-rag", default=False, action="store_true")
    parser.add_argument("--reorder", default=False, action="store_true")
    parser.add_argument("--entity-labels", default=False, action="store_true")
    args = parser.parse_args()
    print("Starting with", args)
    OPENAI_API_KEY = "sk-your-key"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
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
                    n_ctx=8196,
                    temperature=0.1,
                    # draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10),
                )
            else:
                model = Llama.from_pretrained(
                    args.model_spec,
                    filename="*.Q8_0.gguf",
                    n_gpu_layers=-1,
                    n_ctx=8196,
                    temperature=0.1,
                )
    # %%

    # %%

    # %%

    # %%
    data_path = args.data_path
    out_path = Path(args.out_path) / args.out_file
    ner_annotator = NERCleanr(
        langchain=llm,
        model=model,
        gen_tokens=args.gen_tokens,
        add_rag=args.add_rag,
        reorder=args.reorder,
        top_k=args.top_k,
        add_entity_labels=args.entity_labels,
    )
    # annotator = Annotator(model=model, gen_tokens=2048)
    if "articles" in args.data_path:
        eval_set = load_test(data_path)
    else:
        eval_set = load_train(data_path)
        eval_set = {id: article.metadata for id, article in eval_set.items()}
    # few_shot_samples = 10
    # annotator.add_prompt_examples([a for a in eval_set.values()][0:few_shot_samples])

    # %%
    with open("eval_grammar_ner.gbnf", "w") as f:
        f.write(ner_annotator.erl_grammar)
    print(ner_annotator.erl_grammar)

    # %%
    ner_annotator.example_messages

    # %%
    annotations = ner_annotator.annotate(
        {id: article for id, article in list(eval_set.items())}
    )

    output_model = {
        id: {"entities": [e.model_dump() for e in entities]}
        for id, entities in list(annotations.items())
    }
    # %%
    with open(out_path, "w") as f:
        json.dump(output_model, f)
    # %%
