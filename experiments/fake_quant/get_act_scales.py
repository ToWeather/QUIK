#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Author wangsiwen@xiaomi.com
    @file get_act_scales.py
    @Date 2023/12/15 下午2:55
    @Describe 
    @Version 1.0
"""

import json
import torch
import random
from collections import OrderedDict
import modelutils
from llm.model import get_model_and_tokenizer
from llm.data import data_transform

DATA_TRANSFORM_MAP = {
    "zk_sft": data_transform.zk_sft_wh_transform
}


@torch.no_grad()
def get_act_scales(model, dataloader, output_file):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    act_scales = OrderedDict()

    def update_input_info(name):
        def tmp(_, inp, out):
            inp = inp.reshape(-1, inp.shape[-1])
            act_scales.setdefault(name, []).append(inp.abs().mean(0))
        return tmp

    all_handles = []

    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i]
        full = modelutils.find_layers(layer)
        handles = []
        for name in full:
            handles.append(full[name].register_forward_hook(update_input_info(name)))
        all_handles.extend(handles)

    for input_ids in dataloader:
        model(input_ids.to(model.device))

    for handle in all_handles:
        handle.remove()

    model.config.use_cache = use_cache

    for name in act_scales:
        act_scales[name] = torch.stack(act_scales[name]).mean(0)
    torch.save(act_scales, output_file)


def get_calibrate_examples(tokenizer, data_file, sample_cnt, data_transform_type, gist_token, max_seq_len):
    cnt = 0
    candidate_samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cnt += 1
            if cnt > sample_cnt * 10:
                break
            sample = json.loads(line)
            candidate_samples.append(sample)

    random.shuffle(candidate_samples)
    examples = []
    kwargs = {}
    if gist_token:
        kwargs["gist_token"] = gist_token
    for sample in candidate_samples[:sample_cnt]:
        feature = DATA_TRANSFORM_MAP[data_transform_type](sample, **kwargs)
        text = feature["pre_text"] + feature["post_text"]
        input_ids = tokenizer(text)['input_ids']
        if len(input_ids) < max_seq_len:
            input_ids = [tokenizer.pad_token_id] * (max_seq_len - len(input_ids)) + input_ids
        input_ids = input_ids[:max_seq_len]
        examples.append(torch.tensor(input_ids))
    print(f"loaded {len(examples)} samples in total!")
    return examples


def main(model_dir, model_type, data_file, sample_cnt, data_transform_type, max_seq_len, output_file, gist_token=None):
    model, tokenizer = get_model_and_tokenizer(model_dir, model_type)
    model = model.to("cuda:0")
    model.eval()
    dataloader = get_calibrate_examples(tokenizer, data_file, sample_cnt, data_transform_type, gist_token, max_seq_len)
    get_act_scales(model, dataloader, output_file)


if __name__ == "__main__":
    model_dir = "/home/work/wangsiwen/zk_sft/models/train_d15_col_llama2_lr_2e-5_epoch_1"
    model_type = "llama2"
    data_file = "/home/storage00/wangsiwen/zk_sft_corpus/d15/train/zk_mr_wh/train.jsonl"
    sample_cnt = 512
    data_transform_type = "zk_sft"
    max_seq_len = 256
    output_file = "/home/storage00/wangsiwen/code/QUIK/experiments/act_scales/Llama-2-7b-zk-sft.pt"
    main(model_dir, model_type, data_file, sample_cnt, data_transform_type, max_seq_len, output_file)
