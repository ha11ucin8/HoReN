import torch


def parent_module(model, pname):
    components = pname.split(".")
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")
    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")
    return parent


def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")


def _tokenize_prompt_and_label(prompt, label, tokenizer, device):
    if not isinstance(prompt, list):
        prompt = [prompt]
    if not isinstance(label, list):
        label = [label]

    mask_token = -100
    full_prompt = [f"{p} {l}" for p, l in zip(prompt, label)]
    prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]

    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    for i in range(len(prompt)):
        tokens["labels"][i][: num_prompt_toks[i]] = mask_token
    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    return {k: v.to(device) for k, v in tokens.items()}


def tokenize_request(request, tokenizer, device):
    return _tokenize_prompt_and_label(request["prompt"], request["target_new"], tokenizer, device)


def tokenize_unstructured_sample(sample, tokenizer, device):
    answer = sample["answer"]
    if not answer.startswith(" "):
        answer = " " + answer
    return _tokenize_prompt_and_label(sample["question"], answer, tokenizer, device)
