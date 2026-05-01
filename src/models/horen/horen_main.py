from copy import deepcopy
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from .editor import HOREN
from .horen_hparams import HORENHyperParams
from .utils import tokenize_request, tokenize_unstructured_sample


def apply_horen_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: HORENHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)

    editor = HOREN(model=model, config=hparams)
    tokens = tokenize_request(request, tokenizer=tok, device=f"cuda:{hparams.device}")
    editor.edit(tokens=tokens)
    weights_copy = editor.reset_layer
    return editor, weights_copy


def apply_horen_uns_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: HORENHyperParams,
    batch_data: list,
    **kwargs: Any,
):
    """
    Unstructured continual editing API used by examples/run_AKEW_both.py.
    Edits are accumulated into the passed-in model in place.
    """
    editor = HOREN(model=model, config=hparams)
    for sample in batch_data:
        tokens = tokenize_unstructured_sample(sample, tokenizer=tok, device=f"cuda:{hparams.device}")
        editor.edit(tokens=tokens)
    return {}
