"""
Unified entry point for sequential model-editing experiments.

Supported editors : MEND, DEFER, ROME, GRACE, WISE, AlphaEdit, HOREN, ULTRAEDIT
Supported datasets: ZsRE, WikiBigEdit

Usage examples
--------------
# ZsRE with HOREN on LLaMA-3-8B (sequential)
PYTHONPATH=$(pwd) python run_structured_data_editing.py \
    --editing_method HOREN \
    --hparams_dir ./hparams/HOREN/llama3-8b.yaml \
    --data_dir ./data/ZsRE \
    --data_type ZsRE \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit

# WikiBigEdit with WISE on Qwen2.5-7B (sequential)
PYTHONPATH=$(pwd) python run_structured_data_editing.py \
    --editing_method WISE \
    --hparams_dir ./hparams/WISE/qwen2.5-7b.yaml \
    --data_dir ./data/WikiBigEdit/wiki_big_edit.json \
    --data_type WikiBigEdit \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit
"""

from datetime import datetime, timezone
import os
import json
import argparse
import random

import numpy as np
import torch

from src import (
    MENDHyperParams,
    DeferHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    GraceHyperParams,
    WISEHyperParams,
    AlphaEditHyperParams,
    HORENHyperParams,
    UltraEditHyperParams,
    BaseEditor,
    WikiBigEditDataset,
)

HPARAMS_MAP = {
    "MEND": MENDHyperParams,
    "DEFER": DeferHyperParams,
    "ROME": ROMEHyperParams,
    "MEMIT": MEMITHyperParams,
    "GRACE": GraceHyperParams,
    "WISE": WISEHyperParams,
    "AlphaEdit": AlphaEditHyperParams,
    "HOREN": HORENHyperParams,
    "ULTRAEDIT": UltraEditHyperParams,
}


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_zsre(data_dir: str, ds_size: int):
    """Load ZsRE edit and locality data from the given directory."""
    edit_path = os.path.join(data_dir, "zsre_edit_data.json")
    train_path = os.path.join(data_dir, "zsre_train_data.json")

    with open(edit_path, "r", encoding="utf-8") as f:
        edit_raw = json.load(f)
    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)

    edit_data = edit_raw[:ds_size]
    train_data = train_raw[:ds_size]

    prompts = [d["src"] for d in edit_data]
    subject = [d["subject"] for d in edit_data]
    rephrase_prompts = [d["rephrase"] for d in edit_data]
    target_new = [d["answers"][0] for d in edit_data]
    locality_prompts = [d["loc"] for d in train_data]
    locality_ans = [d["loc_ans"] for d in train_data]

    locality_inputs = {
        "neighborhood": {"prompt": locality_prompts, "ground_truth": locality_ans},
    }
    return prompts, subject, rephrase_prompts, target_new, locality_inputs, locality_prompts


def load_wikibigedit(data_path: str, ds_size: int):
    """Load WikiBigEdit data from a JSON file path."""
    datas = WikiBigEditDataset(data_path, size=ds_size)

    prompts = [d["prompt"] for d in datas]
    subject = [d["subject"] for d in datas]
    rephrase_prompts = [d["rephrase"] for d in datas]
    target_new = [d["target_new"] for d in datas]

    locality_prompts = [[d["locality"]] for d in datas]
    locality_ans = [[d["locality_ans"]] for d in datas]
    portability_personas_prompts = [
        [d["portability_personas"]] if isinstance(d["portability_personas"], str) else None for d in datas
    ]
    portability_personas_answers = [[d["target_new"]] for d in datas]
    portability_hop_prompts = [[d["portability_hop"]] if isinstance(d["portability_hop"], str) else None for d in datas]
    portability_hop_answers = [
        [d["portability_hop_ans"]] if isinstance(d["portability_hop_ans"], str) else None for d in datas
    ]

    locality_inputs = {"locality": {"prompt": locality_prompts, "ground_truth": locality_ans}}
    portability_inputs = {
        "personas": {"prompt": portability_personas_prompts, "ground_truth": portability_personas_answers},
        "mhop": {"prompt": portability_hop_prompts, "ground_truth": portability_hop_answers},
    }
    return prompts, subject, rephrase_prompts, target_new, locality_inputs, portability_inputs


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Structured knowledge-editing experiments.")
    parser.add_argument("--editing_method", required=True, type=str, choices=list(HPARAMS_MAP.keys()))
    parser.add_argument("--hparams_dir", required=True, type=str, help="Path to the hparams YAML file.")
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help=(
            "For ZsRE: directory containing zsre_edit_data.json and zsre_train_data.json. "
            "For WikiBigEdit: path to the JSON file (e.g. ./data/WikiBigEdit/wiki_big_edit.json)."
        ),
    )
    parser.add_argument("--data_type", required=True, type=str, choices=["ZsRE", "WikiBigEdit"])
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--ds_size", default=100, type=int)
    parser.add_argument("--sequential_edit", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Load the model in bfloat16. Overrides the bf16 setting in the hparams YAML.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--note", default="", type=str, help="Note appended to output/log filenames.")

    # HOREN-specific overrides
    parser.add_argument(
        "--hopfield_threshold", default=None, type=float, help="(HOREN) Override hopfield_key_match_threshold."
    )
    parser.add_argument(
        "--normalized",
        default=None,
        type=str,
        choices=["true", "false"],
        help="(HOREN) Override normalize_codebook_keys.",
    )
    parser.add_argument(
        "--query_selection_strategy", default=None, type=str, help="(HOREN) Override query_selection_strategy."
    )
    parser.add_argument(
        "--hopfield_max_iter", default=None, type=int, help="(HOREN) Override hopfield_retrieval_max_iter."
    )
    parser.add_argument(
        "--adapter_mode",
        default=None,
        type=str,
        choices=["value", "lora", "none"],
        help="(HOREN) Override adapter_mode.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    editing_hparams = HPARAMS_MAP[args.editing_method]
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    if args.bf16:
        hparams.bf16 = True

    # Apply HOREN-specific CLI overrides
    if args.editing_method == "HOREN":
        if args.hopfield_threshold is not None:
            hparams.hopfield_key_match_threshold = args.hopfield_threshold
        if args.normalized is not None:
            hparams.normalize_codebook_keys = args.normalized.lower() == "true"
        if args.query_selection_strategy is not None:
            hparams.query_selection_strategy = args.query_selection_strategy
        if args.hopfield_max_iter is not None:
            hparams.hopfield_retrieval_max_iter = args.hopfield_max_iter
        if args.adapter_mode is not None:
            hparams.adapter_mode = args.adapter_mode

    model_tag = hparams.model_name.split("/")[-1]
    note_suffix = f"_{args.note}" if args.note else ""

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.data_type}_{model_tag}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}_{timestamp}{note_suffix}.json",
    )
    log_file_prefix = (
        f"{args.data_type}_{model_tag}_{args.editing_method}"
        f"_N={args.ds_size}_Sequential={args.sequential_edit}_{timestamp}{note_suffix}"
    )

    print(f"Results will be saved to: {output_file}")

    editor = BaseEditor.from_hparams(hparams)

    if args.data_type == "ZsRE":
        prompts, subject, rephrase_prompts, target_new, locality_inputs, loc_prompts = load_zsre(
            args.data_dir, args.ds_size
        )
        metrics, _, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            loc_prompts=loc_prompts,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit,
            eval_metric="token em",
            log_file_prefix=log_file_prefix,
            total_ds_size=args.ds_size,
        )
    elif args.data_type == "WikiBigEdit":
        prompts, subject, rephrase_prompts, target_new, locality_inputs, portability_inputs = load_wikibigedit(
            args.data_dir, args.ds_size
        )
        metrics, _, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit,
            eval_metric="token em",
            log_file_prefix=log_file_prefix,
            total_ds_size=args.ds_size,
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved {len(metrics)} records to {output_file}")


if __name__ == "__main__":
    main()
