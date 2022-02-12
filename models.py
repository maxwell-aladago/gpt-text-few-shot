# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = [
    'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
]


class MLPHead(nn.Module):
    def __init__(self, in_chans, rep_size):
        super().__init__()
        self.proj = nn.Linear(in_features=in_chans, out_features=rep_size)

        trunc_normal_(self.proj.weight.data, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, base_model, mlp_head):
        super().__init__()
        self.base = base_model
        self.head = mlp_head

    def forward(self, x, return_base_features=False):
        x = self.base(x)
        if return_base_features:
            return x
        else:
            return self.head(x)


@register_model
def vit_tiny_patch16_224(rep_size=False, **kwargs):
    base = VisionTransformer(
        patch_size=8, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    head = MLPHead(in_chans=192, rep_size=rep_size)
    model = Transformer(base, head)
    return model


@register_model
def vit_small_patch16_224(rep_size=512, **kwargs):
    base = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    head = MLPHead(in_chans=384, rep_size=rep_size)
    model = Transformer(base, head)
    return model


@register_model
def vit_base_patch16_224(rep_size=512, **kwargs):
    base = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    head = MLPHead(in_chans=768, rep_size=rep_size)
    model = Transformer(base, head)
    return model


# @register_model
# def vit_base_patch16_384(rep_size=512, **kwargs):
#     model = VisionTransformer(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#
#     return model

