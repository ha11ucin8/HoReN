from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class HORENHyperParams(HyperParams):
    # Core
    alg_name: str
    model_name: str
    device: int
    inner_params: List[str]

    # Edit loop
    n_iter: int = 20
    edit_lr: float = 1e-2
    lora_edit_lr: float = 5e-3

    # Adapter behavior
    name: str = "Hopfield"
    adapter_mode: str = "value"  # value | lora | none
    val_init: str = "warm"  # warm | cold
    replacement: str = "replace_last"
    eps: float = 0.1
    dist_fn: str = "euc"
    eps_expand: str = "coverage"
    num_pert: str = "1"
    normalize_codebook_keys: bool = False
    query_selection_strategy: str = "last_prompt_token"

    # Hopfield / retrieval
    hopfield_retrieval_beta: float = 1.0
    hopfield_retrieval_eps: float = 1e-5
    hopfield_retrieval_max_iter: int = 8
    hopfield_retrieval_alpha: float = 1.0
    hopfield_key_match_threshold: float = 0.95

    # LoRA adapter mode
    lora_rank: int = 4
    lora_scale: float = 1.0

    # Defaults
    batch_size: int = 1
    max_length: int = 64
    model_parallel: bool = False
    bf16: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "HOREN") or print(
            f"HORENHyperParams can not load from {hparams_name_or_path}, " f'alg_name is {config["alg_name"]}'
        )
        return cls(**config)

    @classmethod
    def from_json(cls, hparams_name_or_path: str):
        # Convenience: allow yaml/json for unstructured runner.
        if hparams_name_or_path.endswith(".yaml") or hparams_name_or_path.endswith(".yml"):
            return cls.from_hparams(hparams_name_or_path)
        return super().from_json(hparams_name_or_path)
