#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Author wangsiwen@xiaomi.com
    @file llama_ft.py
    @Date 2023/12/18 下午2:11
    @Describe 
    @Version 1.0
"""

import argparse
import logging
import os.path

import datautils
import modelutils
import torch
import time
import quik_utils
import quant_sim
import qlinear
import tqdm
import math
import random
import json

from llm.data.field_tune_eval_dataset import FieldTuneDataLoader
from llm.model import get_model_and_tokenizer

random.seed(42)



DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def llama_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', '-m', type=str, required=True,
                        help="the model directory for quantization.")
    parser.add_argument('--model_type', '-t', type=str, required=True,
                        help="the model type.")
    parser.add_argument('--data_file', '-d', type=str, required=True,
                        help="the data file path.")
    parser.add_argument('--max_seq_len', '-l', type=int, default=256,
                        help="the max seq len.")
    parser.add_argument('--sample_cnt', '-s', type=int, default=128,
                        help="the sample count for quantization.")
    parser.add_argument('--act_scale_file_name', type=str, default='Llama-2-7b-hf')
    parser.add_argument('--fp_features', type=int, default=0, help='Number of features to keep in FP16.')
    parser.add_argument('--fp_relative', action='store_true',
                        help='Use relative features for number of fp_features (larger layers have more fp_features)')
    # Act. Quantization Params:
    parser.add_argument('--a_bits', type=int, default=16, choices=[4, 8, 16])

    # Weight Quantization Params:
    parser.add_argument('--w_bits', type=int, default=16, choices=[4, 8, 16])

    parser.add_argument('--int8_down_proj', action='store_true', help='Use INT8 for Down Projection')

    parser.add_argument('--is_quant_model', action="store_true")

    args = parser.parse_args()

    return args


def get_fp_features_num(module: torch.nn.Linear, model, args):
    fp_features_num = args.fp_features
    if args.fp_relative:
        fp_features_num = int(module.in_features / model.config.hidden_size) * args.fp_features
    return fp_features_num


def load_model(args):
    """
    Load model and tokenizer.

    """
    model, tokenizer = get_model_and_tokenizer(args.model_dir, args.model_type)
    model = model.half()
    model = model.to("cuda:0")
    if args.is_quant_model:
        act_scales = torch.load(model.config.act_scale_path)
        weight_scales = torch.load(os.path.join(args.model_dir, 'weight_scales.pt'))

        quant_sim.add_actquant(model)
        layers = modelutils.find_layers(model)

        for name in layers:

            bits = args.a_bits
            if 'lm_head' in name or "rotary_emb" in name:
                print(f'Skipping {name}\n')
                continue

            if 'down_proj' in name:
                if args.int8_down_proj:
                    bits = 8

            if args.fp_features > 0:
                fp_features_num = get_fp_features_num(layers[name].module, model, args)
                if "qkv" in name:
                    act_name = name.replace("qkv", "q")
                else:
                    act_name = name
                layers[name].fp_features_configure(act_scales[act_name], fp_features_num)
            layers[name].quantizer.configure(bits=bits)
        llama_replace_with_kernels(model, act_scales, weight_scales)
    torch.cuda.empty_cache()
    return model, tokenizer


def get_test_samples(tokenizer, test_data_file, sample_cnt, max_seq_len, model_type):
    """
    Get test samples.

    """
    logger = logging.getLogger()
    data_loader = FieldTuneDataLoader(
        model_type, logger, None, test_data_file, tokenizer,
        batch_size=1,
        max_length=max_seq_len,
        eval_type='zk_sft').build_data_loader()
    test_samples = list(data_loader)[:sample_cnt]
    return test_samples


def test_perf(args):
    """
    Test performance.

    """
    print(f"test perf for model {args.model_dir} ...")
    model, tokenizer = load_model(args)
    test_samples = get_test_samples(tokenizer, args.data_file, args.sample_cnt, args.max_seq_len, args.model_type)

    model.config.use_cache = True
    torch.cuda.synchronize()

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    print("test perf for prefill phase...")
    with torch.no_grad():
        times = []
        tknps = []
        torch.cuda.cudart().cudaProfilerStart()

        for i in tqdm.tqdm(range(len(test_samples)), desc='Benchmarking', ncols=80):
            batch = test_samples[i]
            start_time = time.perf_counter()
            out = model(
                **batch.to(model.device)
            )
            sync()
            times.append(time.perf_counter() - start_time)
            tknps.append(batch["input_ids"].shape[-1] / times[-1])
            del out
        sync()
        torch.cuda.cudart().cudaProfilerStop()
        import numpy as np
        print(f'Median times: {np.median(times)} +- {1.96 * np.std(times[2:-2])}')
        print(f'Median tokens/second: {np.median(tknps)} +- {1.96 * np.std(tknps[2:-2])}', )

    print("test perf for decode phase...")
    with torch.no_grad():
        all_times = []
        all_tknps = []

        decode_times = []
        decode_tknps = []
        torch.cuda.cudart().cudaProfilerStart()

        for i in tqdm.tqdm(range(len(test_samples)), desc='Benchmarking', ncols=80):
            batch = test_samples[i]
            start_time = time.perf_counter()
            input_length = batch["input_ids"].shape[1]
            outputs = model.generate(**batch.to(model.device), max_new_tokens=10)
            new_gen_length = outputs.shape[1] - input_length - 1
            sync()
            all_time = time.perf_counter() - start_time
            all_times.append(all_time)
            all_tknps.append((new_gen_length + 1) / all_time)

            decode_used_time = all_time - times[i]
            if decode_used_time > 0:
                decode_times.append(decode_used_time)
                decode_tknps.append(new_gen_length / decode_used_time)
            del outputs
        sync()
        torch.cuda.cudart().cudaProfilerStop()
        import numpy as np
        print(f'Median times: {np.median(decode_times)} +- {1.96 * np.std(decode_times[2:-2])}')
        print(f'Median tokens/second: {np.median(decode_tknps)} +- {1.96 * np.std(decode_tknps[2:-2])}', )

    print("perf for all phase...")
    print(f'Median times: {np.median(all_times)} +- {1.96 * np.std(all_times[2:-2])}')
    print(f'Median tokens/second: {np.median(all_tknps)} +- {1.96 * np.std(all_tknps[2:-2])}', )


def llama_replace_with_kernels(model, act_scales, weight_scales):
    layers = model.model.layers
    shared_inputs = {}

    print("Replace with INT4 kernels.")
    for i in range(len(layers)):
        opt_block = layers[i]
        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]
        full = modelutils.find_layers(opt_block)
        for j, layer_group in enumerate(sequential):
            subset = {n: full[n] for n in layer_group}
            shared_inputs[f"{i}.{j}"] = qlinear.SharedQuantizedInput(len(layer_group))
            for name in subset:
                layer = subset[name]
                if 'lm_head' in name or 'rotary_emb' in name:
                    continue
                is_quantized = False
                bits = 16
                fp_features = 0
                if isinstance(layer, quant_sim.ActQuantWrapper):
                    if layer.quantizer.configured:
                        is_quantized = True
                        bits = layer.quantizer.bits
                        fp_features = layer.fp_features_num
                    layer = layer.module
                layer_weight = layer.weight.data

                layer_scale = weight_scales['model.layers.{}.{}.scale'.format(i, name)]
                if fp_features == 0:
                    fp_feature_idx = None
                else:
                    layer_act_scales = act_scales['model.layers.{}.{}'.format(i, name)]
                    fp_feature_idx = torch.sort(layer_act_scales)[1][-fp_features:]

                if is_quantized:
                    int_mod = qlinear.MixedQLinear.from_float(layer, layer_weight, layer_scale,
                                                              shared_inputs[f"{i}.{j}"], fp_feature_idx,
                                                              bits=bits)
                else:
                    int_mod = layer
                modelutils.replace_single_mod_opt(opt_block, name, int_mod)


if __name__ == '__main__':
    args = llama_parser()

    print(args)

    test_perf(args)
