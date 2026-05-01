import copy
import math
import re
from typing import Any

import torch
import torch.nn.functional as F
import transformers

from time import time

from .utils import brackets_to_periods, parent_module
from ...util.edit_timer import add_edit_time


def _cfg(config: Any, key: str, default=None):
    return getattr(config, key, default)


class HOREN(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.device = f"cuda:{config.device}"
        self.model = model
        self.original_layer = None
        self.edit_log = {}

        for _, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        layer = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)

        if type(original_layer) is not HopfieldAdapter:
            setattr(
                edit_module,
                layer_name,
                HopfieldAdapter(config, original_layer, transpose=transpose).to(self.device),
            )
            self.original_layer = copy.deepcopy(original_layer)
        else:
            # Adapter already installed from a prior edit.  The freeze loop above
            # just set requires_grad=False on all model params, including any
            # lora_A/lora_B/values that add_key() had previously made trainable.
            # Restore their trainability so the optimizer can update them.
            original_layer.activate_params()

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def get_codebook_size(self):
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        adapter = getattr(edit_module, layer_name)
        return len(adapter.keys)

    def generate(self, *args, **kwargs):
        if "input_ids" in kwargs:
            key_id = kwargs["input_ids"].shape[1] - 1
        else:
            key_id = -1
        setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        return self.model.generate(*args, **kwargs)

    def reset_layer(self):
        if self.original_layer is None:
            return
        layer_name = self.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        setattr(edit_module, layer_name, self.original_layer.to(self.device))

    def edit(self, tokens):
        key_id = int((tokens["labels"] == -100).sum(dim=1).min().item() - 1)
        adapter = eval(f"self.model.{self.layer}")
        setattr(adapter, "key_id", key_id)
        setattr(adapter, "training", True)
        setattr(adapter, "edit_label", tokens["labels"])

        lr = (
            _cfg(self.config, "lora_edit_lr", _cfg(self.config, "edit_lr", 1e-2))
            if _cfg(self.config, "adapter_mode", "value") == "lora"
            else _cfg(self.config, "edit_lr", 1e-2)
        )

        self.losses = []
        optimizer = None
        _edit_start = time()
        for i in range(_cfg(self.config, "n_iter", 20)):
            setattr(adapter, "iter", i)
            outputs = self.model(**tokens)
            if i == 0:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                if len(trainable_params) == 0:
                    raise RuntimeError("No trainable parameters found during HOREN edit.")
                optimizer = torch.optim.Adam(trainable_params, lr)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.losses.append(float(loss.detach().cpu().item()))
        add_edit_time(time() - _edit_start)

        setattr(adapter, "training", False)
        chosen_key = getattr(adapter, "chosen_key")
        nkeys = len(getattr(adapter, "keys"))
        self.edit_log["chosen_key"] = int(chosen_key.item())
        self.edit_log["nkeys"] = int(nkeys)
        self.edit_log["key_id"] = key_id

    def debug(self, val=False):
        setattr(eval(f"self.model.{self.layer}"), "debug", val)

    def get_debug_log(self):
        return getattr(eval(f"self.model.{self.layer}"), "debug_log", {})


class HopfieldAdapter(torch.nn.Module):
    _PERC_RE = re.compile(r"^last_(\d+(?:\.\d+)?)_perc_prompt_tokens_avg$")
    _LASTN_RE = re.compile(r"^last_(\d+)_prompt_tokens_avg$")

    def __init__(self, config, layer, transpose):
        super().__init__()
        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.debug = False
        self.config = config

        self.val_init = _cfg(config, "val_init", "warm")
        self.normalize_codebook_keys = bool(_cfg(config, "normalize_codebook_keys", False))
        self.query_selection_strategy = _cfg(config, "query_selection_strategy", "last_prompt_token")
        self.adapter_mode = str(_cfg(config, "adapter_mode", "value")).lower()
        if self.adapter_mode not in {"value", "lora", "none"}:
            raise ValueError("adapter_mode must be one of {'value','lora','none'}")

        self.lora_rank = int(_cfg(config, "lora_rank", 4))
        self.lora_scale = float(_cfg(config, "lora_scale", 1.0))
        self.hopfield_retrieval_beta = float(_cfg(config, "hopfield_retrieval_beta", 1.0))
        self.hopfield_retrieval_eps = float(_cfg(config, "hopfield_retrieval_eps", 1e-5))
        self.hopfield_retrieval_max_iter = int(_cfg(config, "hopfield_retrieval_max_iter", 8))
        self.hopfield_retrieval_alpha = float(_cfg(config, "hopfield_retrieval_alpha", 1.0))
        self.hopfield_key_match_threshold = float(_cfg(config, "hopfield_key_match_threshold", 0.95))

        self.key_id = -1
        self.training = False
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]

        default_key = torch.randn(1, self.key_shape, device=self.device, dtype=layer.weight.dtype)
        if self.normalize_codebook_keys:
            default_key = F.normalize(default_key, p=2, dim=-1)
        self.register_buffer("keys", default_key)

        if self.adapter_mode == "value":
            self.values = torch.nn.Parameter(
                torch.zeros(1, self.value_shape, device=self.device, dtype=torch.float32),
                requires_grad=False,
            )
            self.lora_A = None
            self.lora_B = None
        elif self.adapter_mode == "lora":
            self.values = None
            r = self.lora_rank
            self.lora_A = torch.nn.Parameter(
                torch.zeros(1, r, self.value_shape, device=self.device, dtype=torch.float32),
                requires_grad=False,
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(1, self.value_shape, r, device=self.device, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.values = None
            self.lora_A = None
            self.lora_B = None
        self.key_labels = [torch.tensor(-1, device=self.device)]
        self.debug_log = {}

    def _select_query(self, layer_input: torch.Tensor, last_prompt_token_index: int):
        strategy = self.query_selection_strategy
        if strategy == "last_prompt_token" or last_prompt_token_index == -1:
            return layer_input[:, last_prompt_token_index, :]
        if strategy == "first_prompt_token":
            return layer_input[:, 0, :]

        prompt_token_len = last_prompt_token_index + 1
        m = self._PERC_RE.match(strategy)
        if m:
            perc = float(m.group(1)) / 100.0
            if not (0.0 < perc <= 1.0):
                raise ValueError(f"Invalid query_selection_strategy={strategy}")
            k = max(int(math.ceil(prompt_token_len * perc)), 1)
            return layer_input[:, prompt_token_len - k : prompt_token_len, :].mean(dim=1)

        m = self._LASTN_RE.match(strategy)
        if m:
            n = int(m.group(1))
            if n <= 0:
                raise ValueError(f"Invalid query_selection_strategy={strategy}")
            k = min(n, prompt_token_len)
            return layer_input[:, prompt_token_len - k : prompt_token_len, :].mean(dim=1)

        raise ValueError(f"Invalid query_selection_strategy={strategy}")

    def _query(self, q: torch.Tensor):
        if q.ndim != 2:
            raise ValueError(f"q must be [B,D], got {tuple(q.shape)}")
        K = self.keys.to(device=q.device, dtype=q.dtype)
        qt = q
        for _ in range(self.hopfield_retrieval_max_iter):
            scores = self.hopfield_retrieval_beta * (qt @ K.t())
            probs = F.softmax(scores, dim=-1)
            retrieved = probs @ K
            if self.normalize_codebook_keys:
                retrieved = F.normalize(retrieved, p=2, dim=-1)
            if (retrieved - qt).norm(dim=-1).max().item() < self.hopfield_retrieval_eps:
                break
            qt = (1.0 - self.hopfield_retrieval_alpha) * qt + self.hopfield_retrieval_alpha * retrieved
            if self.normalize_codebook_keys:
                qt = F.normalize(qt, p=2, dim=-1)
        return qt @ K.t()

    def _apply_lora(self, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, match_mask: torch.Tensor):
        if a.ndim == 2:
            a = a.unsqueeze(0)
        if b.ndim == 2:
            b = b.unsqueeze(0)
        y = y.to(dtype=a.dtype, device=a.device)
        gate = match_mask.to(device=y.device, dtype=y.dtype).view(-1, 1)
        # y: [B, D], a: [B, r, D], b: [B, D, r]
        # Use bmm to avoid matmul's batch-broadcasting producing [B, B, r].
        tmp = torch.bmm(y.unsqueeze(1), a.transpose(-1, -2)).squeeze(1)  # [B, r]
        delta = torch.bmm(tmp.unsqueeze(1), b.transpose(-1, -2)).squeeze(1)  # [B, D]
        return (self.lora_scale * gate) * delta

    def add_key(self, new_key, layer_out, token_idx):
        self.keys = torch.cat([self.keys, new_key.detach().to(self.keys.dtype)], dim=0)

        if self.adapter_mode == "value":
            if self.val_init == "cold":
                new_val = torch.rand(1, self.value_shape, device=self.device, dtype=torch.float32)
            else:
                new_val = layer_out[:, token_idx, :].detach().to(torch.float32)
            self.values = torch.nn.Parameter(torch.cat([self.values, new_val], dim=0), requires_grad=True)

        if self.adapter_mode == "lora":
            r = self.lora_rank
            new_a = 0.01 * torch.randn(1, r, self.value_shape, device=self.device, dtype=torch.float32)
            new_b = torch.zeros(1, self.value_shape, r, device=self.device, dtype=torch.float32)
            self.lora_A = torch.nn.Parameter(torch.cat([self.lora_A, new_a], dim=0), requires_grad=True)
            self.lora_B = torch.nn.Parameter(torch.cat([self.lora_B, new_b], dim=0), requires_grad=True)

        self.key_labels.append(getattr(self, "edit_label", torch.tensor([-1], device=self.device)))

    def activate_params(self):
        """Restore requires_grad=True on all adapter parameters that were populated
        by add_key() (i.e. everything except the zero-init default slot at index 0).
        Called by HOREN.__init__ on every edit after the first, because the freeze
        loop in __init__ unconditionally sets requires_grad=False on all model params,
        which would otherwise prevent training when an existing key is reused."""
        if self.adapter_mode == "value" and isinstance(self.values, torch.nn.Parameter):
            self.values.requires_grad_(True)
        elif self.adapter_mode == "lora":
            if isinstance(self.lora_A, torch.nn.Parameter):
                self.lora_A.requires_grad_(True)
            if isinstance(self.lora_B, torch.nn.Parameter):
                self.lora_B.requires_grad_(True)

    def label_match(self, edit_label, key_label):
        if key_label.numel() == 1 and key_label.item() == -1:
            return False
        return edit_label.float().mean() == key_label.float().mean()

    def forward(self, *args):
        layer_out = self.layer(*args)
        if self.adapter_mode == "none":
            return layer_out
        if not self.training and len(self.keys) == 1:
            return layer_out

        last_prompt_token_index = min(self.key_id, args[0].shape[1] - 1)
        query = self._select_query(args[0], last_prompt_token_index)
        if self.normalize_codebook_keys:
            query = F.normalize(query, p=2, dim=-1)

        matching_scores = self._query(query)
        if self.training and getattr(self, "iter", 0) == 0:
            max_score, chosen_key = torch.max(matching_scores, dim=-1)
            should_add_key = False
            if max_score.item() > self.hopfield_key_match_threshold:
                nearest_label = self.key_labels[int(chosen_key.item())]
                if not self.label_match(self.edit_label, nearest_label):
                    should_add_key = True
            else:
                should_add_key = True
            if should_add_key:
                self.add_key(query, layer_out, last_prompt_token_index)
                matching_scores = self._query(query)

        max_score, self.chosen_key = torch.max(matching_scores, dim=-1)
        is_match = max_score > self.hopfield_key_match_threshold

        if self.adapter_mode == "value":
            chosen_values = self.values[self.chosen_key]
            layer_update = chosen_values * is_match.unsqueeze(1).to(chosen_values.dtype)
            layer_out = layer_out.clone()
            layer_out[:, last_prompt_token_index, :] += layer_update.to(layer_out.dtype)
        elif self.adapter_mode == "lora":
            a = self.lora_A[self.chosen_key]
            b = self.lora_B[self.chosen_key]
            layer_update = self._apply_lora(y=layer_out[:, last_prompt_token_index, :], a=a, b=b, match_mask=is_match)
            layer_out = layer_out.clone()
            layer_out[:, last_prompt_token_index, :] += layer_update.to(layer_out.dtype)
        return layer_out
