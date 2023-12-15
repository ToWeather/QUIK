import argparse
import datautils
import modelutils
import torch
import quik_utils
import quant
import sparseGPT_utils
import json
import os
import random
from llm.data import data_transform
from llm.model import get_model_and_tokenizer

random.seed(42)

DATA_TRANSFORM_MAP = {
    "zk_sft": data_transform.zk_sft_wh_transform
}

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def llama_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', '-m', type=str, required=True,
                        help="the model directory for quantization.")
    parser.add_argument('--model_type', '-t', type=str, required=True,
                        help="the model type.")
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help="the output model directory.")
    parser.add_argument('--data_file', '-d', type=str, required=True,
                        help="the data file path.")
    parser.add_argument('--transform_type', '-f', type=str, required=True,
                        choices=list(DATA_TRANSFORM_MAP),
                        help="the data transform type.")
    parser.add_argument('--sample_cnt', '-s', type=int, default=128,
                        help="the sample count for quantization.")

    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument('--fp_features', type=int, default=0, help='Number of features to keep in FP16.')
    parser.add_argument('--fp_threshold', type=float, default=0.0,
                        help='Threshold where we put the fp features to zero.')
    parser.add_argument('--fp_relative', action='store_true',
                        help='Use relative features for number of fp_features (larger layers have more fp_features)')

    # Act. Quantization Params:
    parser.add_argument('--a_bits', type=int, default=16, choices=[4, 8, 16])

    # Weight Quantization Params:
    parser.add_argument('--w_bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument('--w_clip', action='store_true', help='Use clipping for weight quantization')
    parser.add_argument('--w_asym', action='store_true')

    parser.add_argument('--int8_down_proj', action='store_true', help='Use INT8 for Down Projection')

    parser.add_argument('--gist_token', type=str, default="",
                        help="the gist token.")

    args = parser.parse_args()

    return args


@torch.no_grad()
def llama_sequential(model, dataloader, act_scales, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = modelutils.find_layers(layer)

        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]
        for names in sequential:
            subset = {n: full[n] for n in names}

            modules_quik = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)

                # Extract the number of outliers
                if args.fp_relative:
                    outlier_num = int(subset[name].in_features / model.config.hidden_size) * args.fp_features
                else:
                    outlier_num = args.fp_features

                layer_scales = None
                if outlier_num > 0:
                    layer_scales = act_scales['model.layers.{}.{}'.format(i, name)]
                    max_val = layer_scales.abs().max()
                    fp_threshold = args.fp_threshold

                    if 'down_proj' in name and args.int8_down_proj:
                        fp_threshold *= 2

                    if max_val <= fp_threshold:
                        outlier_num = 0
                        layer_scales = None

                modules_quik[name] = quik_utils.QUIK(
                    layer=subset[name],
                    act_scales=layer_scales,
                    fp_features=outlier_num
                )
                modules_quik[name].quantizer = quant.WeightQuantizer()

                current_w_bits = args.w_bits
                if 'down_proj' in name:
                    if args.int8_down_proj:
                        current_w_bits = 8
                modules_quik[name].quantizer.configure(
                    current_w_bits, perchannel=True, sym=not (args.w_asym), mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    modules_quik[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                modules_quik[name].fasterquant(percdamp=args.percdamp, groupsize=-1)
                quantizers['model.layers.%d.%s' % (i, name)] = modules_quik[name].quantizer
                modules_quik[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del modules_quik
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


def get_calibrate_examples(tokenizer, data_file, sample_cnt, data_transform_type, gist_token):
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
        examples.append((input_ids, input_ids))
    print(f"loaded {len(examples)} samples in total!")
    return examples


if __name__ == '__main__':
    args = llama_parser()

    print(args)

    model, tokenizer = get_model_and_tokenizer(args.model_dir, args.model_type)
    model.eval()

    model.config.w_bits = args.w_bits
    model.config.a_bits = args.a_bits

    model.config.fp_features = args.fp_features
    model.config.fp_relative = args.fp_relative
    model.config.int8_down_proj = args.int8_down_proj
    model.config.fp_threshold = args.fp_threshold

    # Extract Scale
    if args.w_bits < 16 or args.a_bits < 16:
        if args.fp_features > 0:
            relative_path = os.path.join(modelutils.act_scale_dir, "{}.pt".format(args.model.split('/')[-1]))
            model.config.act_scale_path = relative_path
            act_scales = torch.load(relative_path)
            print('Loaded act_scales from: ', relative_path)
        else:
            act_scales = None
            print('No act_scales loaded')

    # Apply GPTQ on the model
    if args.w_bits < 16:
        dataloader = get_calibrate_examples(
            tokenizer, args.data_file, args.sample_cnt, args.data_transform_type, args.gist_token)
        quantizers = llama_sequential(model, dataloader, act_scales, DEV, args)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # # Add Input Quantization
    # if args.a_bits < 16:
    #     number_of_zero_outlier_linear = 0
    #     print('.....Activation Quantization.....')
    #     quant.add_actquant(model)
    #     layers = modelutils.find_layers(model)
    #
    #     for name in layers:
    #
    #         # Skip lm_head quantization
    #         if 'lm_head' in name:
    #             print(f'Skipping {name}\n')
    #             continue
    #
    #         current_a_bits = args.a_bits
    #
    #         # Extract the number of outliers
    #         if args.fp_relative:
    #             outlier_num = int(layers[name].module.in_features / model.config.hidden_size) * args.fp_features
    #         else:
    #             outlier_num = args.fp_features
    #
    #         fp_threshold = args.fp_threshold
    #         if 'down_proj' in name and args.int8_down_proj:
    #             fp_threshold *= 2
    #             current_a_bits = 8
    #
    #         if outlier_num > 0 and 'lm_head' not in name:
    #             max_val = act_scales[name].abs().max()
    #             if max_val > fp_threshold:
    #                 layers[name].fp_features_configure(act_scales[name], outlier_num)
    #             else:
    #                 layers[name].fp_features_configure(act_scales[name], 0)
    #                 number_of_zero_outlier_linear += 1
    #
    #         print(f'{name}: {outlier_num} outliers - {current_a_bits} bits', flush=True)
    #         layers[name].quantizer.configure(bits=current_a_bits)
    #
    #     print(f'{number_of_zero_outlier_linear} layers with zero outliers.\n')
