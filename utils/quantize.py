"""Post-training W4A16 NF4 quantization via bitsandbytes.
Recursively replace all nn.Linear layers of the model will be quantized in-place with 4-bit NF4 quantized layers.
"""

import bitsandbytes.nn as bnb_nn
import torch
import torch.nn as nn


def quantize_linear_layers(module):
    for child_name, child in module.named_children():
        if isinstance(child, nn.Linear):
            quantized_layer = bnb_nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                quant_type="nf4",
                compute_dtype=torch.float16,
            )
            quantized_layer.weight.data = child.weight.data
            if child.bias is not None:
                quantized_layer.bias.data = child.bias.data
            setattr(module, child_name, quantized_layer)
        else:
            quantize_linear_layers(child)
