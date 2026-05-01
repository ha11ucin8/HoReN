"""
Unstructured knowledge-editing experiments.

Supported editors : HOREN, UNKE
Supported datasets: unke (UnKE unstructured format)

Data directory layout
---------------------
  data/
    UnKE/
      final_data_v3.json      -- edit samples (required)
    alpaca_data.json          -- general instruction data for UNKE locality (required for UNKE only)

Usage examples
--------------
# UnKE on LLaMA-3-8B
PYTHONPATH=$(pwd) python experiments/run_unstructured_data_editing.py \\
    --editing_method UNKE \\
    --hparams_dir ./hparams/UNKE/llama3-8b.json \\
    --data_dir ./data \\
    --data_type unke \\
    --ds_size 100 \\
    --output_dir ./outputs
    --sequential_edit

# HOREN on LLaMA-3-8B
PYTHONPATH=$(pwd) python experiments/run_unstructured_data_editing.py \\
    --editing_method HOREN \\
    --hparams_dir ./hparams/HOREN/llama3-8b.yaml \\
    --data_dir ./data \\
    --data_type unke \\
    --ds_size 100 \\
    --output_dir ./outputs
    --sequential_edit
"""

import os
import json
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.AKEW_both import (
    AKEWUnifiedDataset,
    get_llama_without_answer,
    get_qwen_without_answer,
)
from src.evaluate.evaluate_uns import eval_akew_unstructured
from src.models.horen import HORENHyperParams, apply_horen_uns_to_model
from src.models.unke import unkeHyperParams, apply_unke_to_model


HPARAMS_MAP = {
    "HOREN": HORENHyperParams,
    "UNKE": unkeHyperParams,
}

APPLY_FN = {
    "HOREN": apply_horen_uns_to_model,
    "UNKE": apply_unke_to_model,
}


def set_seed(seed: int = 2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_hparams(editing_method: str, hparams_dir: str):
    cls = HPARAMS_MAP[editing_method]
    if hparams_dir.endswith(".json"):
        return cls.from_json(hparams_dir)
    return cls.from_hparams(hparams_dir)


def load_model_and_tokenizer(model_name: str):
    model_name_lower = model_name.lower()
    use_bf16 = (
        "llama" in model_name_lower
        or "qwen" in model_name_lower
        or "deepseek" in model_name_lower
        or "oss" in model_name_lower
    ) and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def generate_predictions(model, gen_tok, samples, clean_answer_fn):
    """Generate predictions for a list of samples.

    Returns a new list of dicts with ``original_prediction``, ``para_prediction``,
    ``sub_pred``, and cleaned ``answer`` fields added.
    """
    results = []
    for sample in tqdm(samples, desc="  Generating predictions", leave=False):
        data = dict(sample)

        question_texts = [data["question"], data["para_question"]]
        inputs = gen_tok(question_texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=512,
            )
        decoded = gen_tok.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs["input_ids"], out_ids)],
            skip_special_tokens=True,
        )
        data["original_prediction"] = decoded[0]
        data["para_prediction"] = decoded[1]
        data["answer"] = clean_answer_fn(data["answer"])

        if data.get("sub_question"):
            sub_inputs = gen_tok(data["sub_question"], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                sub_out_ids = model.generate(
                    input_ids=sub_inputs["input_ids"],
                    attention_mask=sub_inputs["attention_mask"],
                    do_sample=False,
                    max_new_tokens=512,
                )
            data["sub_pred"] = gen_tok.batch_decode(
                [out[len(inp) :] for inp, out in zip(sub_inputs["input_ids"], sub_out_ids)],
                skip_special_tokens=True,
            )
        else:
            data["sub_pred"] = []

        results.append(data)
    return results


def save_checkpoint(edited_data, checkpoint_n, log_dir, log_prefix, data_type, bert_model_path, device):
    """Write predictions + metrics for *checkpoint_n* samples to ``logs/{log_prefix}_Checkpoint_{n}_results.json``."""
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_file = os.path.join(log_dir, f"{log_prefix}_Checkpoint_{checkpoint_n}_results.json")
    # Write samples first so eval_akew_unstructured can read them
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(edited_data, f, indent=4)
    metrics = eval_akew_unstructured(checkpoint_file, data_type, bert_model_path, device)
    # Overwrite with combined {metrics, samples} in one file
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": edited_data}, f, indent=4)
    print(f"[Checkpoint {checkpoint_n}] Saved to {checkpoint_file}")
    return metrics


def format_alpaca(raw_alpaca: list, model_name: str) -> list:
    model_name_lower = model_name.lower()
    formatted = []
    for item in raw_alpaca:
        text = item["instruction"] + item["input"]
        answer = item["output"]
        if "llama" in model_name_lower:
            formatted.append(get_llama_without_answer(text) + answer)
        elif "qwen" in model_name_lower or "deepseek" in model_name_lower:
            formatted.append(get_qwen_without_answer(text) + answer)
        else:
            formatted.append(text + " " + answer)
    return formatted


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Unstructured knowledge-editing experiments.")
    parser.add_argument("--editing_method", required=True, type=str, choices=list(HPARAMS_MAP.keys()))
    parser.add_argument(
        "--hparams_dir", required=True, type=str, help="Path to hparams file (.json for UNKE, .yaml for HOREN)."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help=(
            "Parent data directory. AKEWUnifiedDataset looks for <data_dir>/UnKE/final_data_v3.json. "
            "UNKE method also requires <data_dir>/alpaca_data.json for locality regularisation."
        ),
    )
    parser.add_argument("--data_type", required=True, type=str, choices=["unke", "counterfact", "wikiupdate", "mquake"])
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--log_dir", default="./logs", type=str, help="Directory for checkpoint result files.")
    parser.add_argument("--ds_size", default=100, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="Number of edits per batch.")
    parser.add_argument(
        "--bert_model_path",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="SentenceTransformer model path for BERTScore in evaluation.",
    )
    parser.add_argument("--device", default=0, type=int, help="CUDA device index.")
    parser.add_argument("--seed", default=2024, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    hparams = load_hparams(args.editing_method, args.hparams_dir)
    model_tag = Path(hparams.model_name).name
    log_prefix = f"unstructured_{model_tag}_{args.editing_method}" f"_{args.data_type}_N={args.ds_size}_{timestamp}"

    print(f"Loading model: {hparams.model_name}")
    model, tok = load_model_and_tokenizer(hparams.model_name)

    # Dataset — pass model_name so AKEW applies the right chat template
    ds = AKEWUnifiedDataset(
        args.data_dir,
        dataset_type=args.data_type,
        model_name=hparams.model_name,
        size=args.ds_size,
        use_unstructured_data=True,
    )
    print(f"Loaded {len(ds)} samples from {args.data_type}.")

    # Alpaca locality data (UNKE only)
    alpaca_data = None
    if args.editing_method == "UNKE":
        alpaca_path = os.path.join(args.data_dir, "alpaca_data.json")
        if not os.path.exists(alpaca_path):
            raise FileNotFoundError(
                f"alpaca_data.json not found at {alpaca_path}. " "UNKE requires this file for locality preservation."
            )
        with open(alpaca_path, "r", encoding="utf-8") as f:
            raw_alpaca = json.load(f)
        alpaca_data = format_alpaca(raw_alpaca, hparams.model_name)
        print(f"Loaded {len(alpaca_data)} Alpaca samples for UNKE locality regularisation.")

    # ── Generation tokenizer (left-padded for batched generation) ────────────
    gen_tok = AutoTokenizer.from_pretrained(hparams.model_name, padding_side="left")
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    if gen_tok.pad_token_id is None and gen_tok.eos_token_id is not None:
        gen_tok.pad_token_id = gen_tok.eos_token_id

    def clean_answer(answer: str) -> str:
        for suffix in ["<|eot_id|>", "<|im_end|>"]:
            if answer.endswith(suffix):
                return answer[: -len(suffix)]
        return answer

    # ── Checkpoint schedule (mirrors structured-data script) ─────────────────
    _cp_set = {1, 10, 30, 100, 120, 500} if args.ds_size <= 1000 else {5000}
    _metric_period = 1000 if args.ds_size < 4000 else 2000
    _cp_set.add(len(ds))
    for _m in range(_metric_period, len(ds) + 1, _metric_period):
        _cp_set.add(_m)
    checkpoint_periods = sorted(n for n in _cp_set if 1 <= n <= len(ds))

    # ── Apply edits with checkpoint evaluation ────────────────────────────────
    apply_fn = APPLY_FN[args.editing_method]
    num_batches = len(ds) // args.batch_size + (1 if len(ds) % args.batch_size else 0)

    edits_done = 0
    cp_ptr = 0
    last_preds = []

    for batch_idx in tqdm(range(num_batches), desc="Editing"):
        start = batch_idx * args.batch_size
        batch = ds[start : start + args.batch_size]
        if not isinstance(batch, list):
            batch = [batch]

        ex_kwargs = {}
        if args.editing_method == "UNKE" and alpaca_data:
            ex_kwargs["ex_data"] = random.sample(alpaca_data, min(hparams.ex_data_num, len(alpaca_data)))

        apply_fn(model, tok, hparams, batch, **ex_kwargs)
        edits_done = min(start + args.batch_size, len(ds))

        # Fire any checkpoints whose threshold has been reached
        while cp_ptr < len(checkpoint_periods) and edits_done >= checkpoint_periods[cp_ptr]:
            n = checkpoint_periods[cp_ptr]
            print(f"\n=== Checkpoint {n} / {len(ds)} ===")
            checkpoint_preds = generate_predictions(model, gen_tok, ds[0:n], clean_answer)
            save_checkpoint(
                checkpoint_preds,
                n,
                args.log_dir,
                log_prefix,
                args.data_type,
                args.bert_model_path,
                args.device,
            )
            last_preds = checkpoint_preds
            cp_ptr += 1

    # ── Save final output (full dataset) ─────────────────────────────────────
    # last_preds already holds the predictions for the final checkpoint (len(ds)).
    # The checkpoint already wrote the combined file to log_dir; mirror it to output_dir.
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(
        args.output_dir,
        f"unstructured_{model_tag}_{args.editing_method}_{args.data_type}_N={args.ds_size}_{timestamp}.json",
    )
    # Re-evaluate on the final predictions (already computed inside save_checkpoint,
    # but we write to output_dir with the same {metrics, samples} structure).
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(last_preds, f, indent=4)
    final_metrics = eval_akew_unstructured(result_file, args.data_type, args.bert_model_path, args.device)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": final_metrics, "samples": last_preds}, f, indent=4)
    print(f"Saved {len(last_preds)} records + metrics to {result_file}")


if __name__ == "__main__":
    main()
