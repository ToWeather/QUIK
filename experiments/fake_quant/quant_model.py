#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Author wangsiwen@xiaomi.com
    @file quant_model.py
    @Date 2023/12/15 上午10:08
    @Describe 
    @Version 1.0
"""

import torch
import quant
import modelutils
from llm.model import get_model_and_tokenizer


def load_fake_quant_model(model_dir, model_type):
    """
    Load fake quant model.

    """
    model, tokenizer = get_model_and_tokenizer(model_dir, model_type)

    # Add Input Quantization
    if model.config.a_bits < 16:
        number_of_zero_outlier_linear = 0
        print('.....Activation Quantization.....')
        quant.add_actquant(model)
        layers = modelutils.find_layers(model)

        act_scales = torch.load(model.config.act_scale_path)

        for name in layers:

            # Skip lm_head quantization
            if 'lm_head' in name:
                print(f'Skipping {name}\n')
                continue

            current_a_bits = model.config.a_bits

            # Extract the number of outliers
            if model.config.fp_relative:
                outlier_num = int(layers[name].module.in_features / model.config.hidden_size) * model.config.fp_features
            else:
                outlier_num = model.config.fp_features

            fp_threshold = model.config.fp_threshold
            if 'down_proj' in name and model.config.int8_down_proj:
                fp_threshold *= 2
                current_a_bits = 8

            if outlier_num > 0 and 'lm_head' not in name:
                max_val = act_scales[name].abs().max()
                if max_val > fp_threshold:
                    layers[name].fp_features_configure(act_scales[name], outlier_num)
                else:
                    layers[name].fp_features_configure(act_scales[name], 0)
                    number_of_zero_outlier_linear += 1

            print(f'{name}: {outlier_num} outliers - {current_a_bits} bits', flush=True)
            layers[name].quantizer.configure(bits=current_a_bits)

        print(f'{number_of_zero_outlier_linear} layers with zero outliers.\n')

    return model, tokenizer
