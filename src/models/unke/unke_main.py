import torch
import torch.nn as nn
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from ...util import nethook
import torch.optim as optim

try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
except ImportError:
    _prepare_4d_causal_attention_mask = None

from .unke_hparams import unkeHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeHyperParams,
    layer: int,
):
    input_ids = tok(batch_data, padding=True, return_tensors="pt").to("cuda")
    idxs = [i.sum() - 1 for i in input_ids["attention_mask"]]
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = model(**input_ids)
            zs_out = tr.output
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list = []
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i, idxs[i]])
    zs_out = torch.stack(zs_out_list, dim=0)

    return zs_out, idxs


def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
    no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def apply_unke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: unkeHyperParams,
    batch_data: list,
    ex_data: list,
):
    preserve_params = []
    for name, params in model.named_parameters():
        splitted_name = name.split(".")
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    weights = {param: nethook.get_parameter(model, param) for param in preserve_params}
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    z_layer = hparams.layers[-1]
    z_list = []
    for data in batch_data:
        cur_z = compute_z(model, tok, data, z_layer, hparams)
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=0)

    batch_question = [i["question"] for i in batch_data]

    for i, layer in enumerate(hparams.layers):
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input
                layer_out_ks = tr.output
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks

        cur_zs, idxs = compute_ks(model, tok, batch_question, hparams, z_layer)

        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(next(model.parameters()).device)

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out

        resid = targets / (len(hparams.layers) - i)

        criterion = nn.MSELoss()
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))

        for n, m in _layer.named_parameters():
            m.requires_grad = True

        params = get_optimizer_params(_layer, hparams.lr)
        optimizer = optim.AdamW(params, lr=hparams.lr, eps=1e-8, betas=(0.9, 0.999))

        for j in range(len(idxs)):
            layer_out_ks[j, idxs[j]] += resid[j]

        model_name_lower = hparams.model_name.lower()
        if "llama" in model_name_lower:
            input_causal_mask, input_position_ids, input_cache_position = get_causal_mask(
                layer_in_ks, contexts_tok["attention_mask"]
            )
            ex_causal_mask, ex_position_ids, ex_cache_position = get_causal_mask(stat_in, ex_tok["attention_mask"])
        elif "qwen" in model_name_lower or "deepseek" in model_name_lower:
            input_causal_mask, input_position_ids = get_qwen2_causal_mask(layer_in_ks, contexts_tok["attention_mask"])
            ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(stat_in, ex_tok["attention_mask"])

        for step in range(hparams.optim_num_step):
            optimizer.zero_grad()
            if "qwen" in model_name_lower or "deepseek" in model_name_lower:
                loss = criterion(
                    _layer(stat_in, attention_mask=ex_causal_mask, position_ids=ex_position_ids)[0], stat_out
                ) + criterion(
                    _layer(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids)[0],
                    layer_out_ks,
                )
            elif "llama" in model_name_lower:
                llama_backbone = getattr(model, "model", None)
                rotary_emb = getattr(llama_backbone, "rotary_emb", None)
                if rotary_emb is None:
                    raise RuntimeError("Llama backbone rotary embedding module not found.")
                ex_pos_emb = rotary_emb(stat_in, ex_position_ids)
                input_pos_emb = rotary_emb(layer_in_ks, input_position_ids)
                loss = criterion(
                    _layer(
                        stat_in,
                        attention_mask=ex_causal_mask,
                        position_ids=ex_position_ids,
                        cache_position=ex_cache_position,
                        position_embeddings=ex_pos_emb,
                    )[0],
                    stat_out,
                ) + criterion(
                    _layer(
                        layer_in_ks,
                        attention_mask=input_causal_mask,
                        position_ids=input_position_ids,
                        cache_position=input_cache_position,
                        position_embeddings=input_pos_emb,
                    )[0],
                    layer_out_ks,
                )
            else:
                loss = criterion(_layer(stat_in)[0], stat_out) + criterion(_layer(layer_in_ks)[0], layer_out_ks)

            loss.backward(retain_graph=True)
            optimizer.step()

        for x in [layer_in_ks, layer_out_ks, cur_zs, targets, stat_in, stat_out]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    return weights_copy


def get_qwen2_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    if _prepare_4d_causal_attention_mask is None:
        raise RuntimeError(
            "transformers._prepare_4d_causal_attention_mask unavailable in this version. "
            "Install transformers==4.40 or adapt get_qwen2_causal_mask for your version."
        )
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (input_tensor.shape[0], input_tensor.shape[1]),
        input_tensor,
        0,
    )
    return attention_mask, position_ids


def get_causal_mask(input_tensor, attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(dtype)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    cache_position = torch.arange(0, sequence_length, device=device)

    return causal_mask, position_ids, cache_position
