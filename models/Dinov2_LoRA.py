import os
import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file

from utils.misc import initialize_weights
from models.Dino.backbones import DinoV2_self

working_path = os.path.abspath('.')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class _LoRA_qkv(nn.Module):
    """Inject LoRA into DINOv2 SelfAttention.qkv."""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, 3*dim
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class LoRA_Dino(nn.Module):
    def __init__(self, dino_model, r: int, lora_layer: Optional[List[int]] = None):
        super(LoRA_Dino, self).__init__()
        assert r > 0

        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(dino_model.dino_model.blocks)))

        self.w_As = []
        self.w_Bs = []

        # Freeze backbone
        for param in dino_model.dino_model.parameters():
            param.requires_grad = False

        # LoRA surgery
        for t_layer_i, blk in enumerate(dino_model.dino_model.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.dino = dino_model

    def get_image_embeddings(self, x: Tensor) -> Tensor:
        return self.dino(x)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class Dinov2_CD_LoRA(nn.Module):
    """DINOv2 Change Detection with LoRA.
    
    Args:
        num_embed: Output embedding dimension
        model_name: DINOv2 model variant ('dinov2_vitb14' or 'dinov2_vitl14')
        lora_r: LoRA rank
        lora_layers: Which layers to apply LoRA (default: [8,9,10,11])
        heterogeneous: Enable heterogeneous mode with two separate encoders
    """

    def __init__(
        self,
        num_embed: int = 16,
        model_name: str = 'dinov2_vitb14',
        lora_r: int = 4,
        lora_layers: Optional[List[int]] = None,
        heterogeneous: bool = False,
    ):
        super(Dinov2_CD_LoRA, self).__init__()

        self.heterogeneous = heterogeneous
        self._patch_size = 14  # DINOv2 uses patch size 14

        # Determine channel count based on model
        in_channels = 1024 if 'vitl' in model_name else 768

        def _create_dino(model_name: str):
            return DinoV2_self(
                model_name=model_name,
                layer1=11,
                facet1="value",
                use_cls=False,
                norm_descs=True,
                device="cuda",
                pretrained=True,
            )

        if self.heterogeneous:
            print("[dinov2] Heterogeneous mode enabled: using two separate encoders")
            dino1 = _create_dino(model_name)
            dino2 = _create_dino(model_name)

            # Encoder 1: smaller LoRA, fewer layers
            lora_layers_1 = lora_layers if lora_layers is not None else [8, 9, 10, 11]
            self.dino1 = LoRA_Dino(dino1, r=lora_r, lora_layer=lora_layers_1)

            # Encoder 2: larger LoRA, all layers
            depth = len(dino2.dino_model.blocks)
            lora_layers_2 = list(range(depth))
            self.dino2 = LoRA_Dino(dino2, r=lora_r * 8, lora_layer=lora_layers_2)

            # Two separate adapters
            self.Adapter1 = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, num_embed, kernel_size=1),
            )
            self.Adapter2 = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, num_embed, kernel_size=1),
            )
            initialize_weights(self.Adapter1, self.Adapter2)
        else:
            dino = _create_dino(model_name)
            lora_layers_default = lora_layers if lora_layers is not None else [8, 9, 10, 11]
            self.dino = LoRA_Dino(dino, r=lora_r, lora_layer=lora_layers_default)

            self.Adapter = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, num_embed, kernel_size=1),
            )
            initialize_weights(self.Adapter)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _pad_to_patch_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad inputs to multiples of the patch size (14) when necessary."""
        ps = self._patch_size
        if (x.shape[-2] % ps) or (x.shape[-1] % ps):
            new_h = ps * (x.shape[-2] // ps + (1 if x.shape[-2] % ps else 0))
            new_w = ps * (x.shape[-1] // ps + (1 if x.shape[-1] % ps else 0))
            x = F.interpolate(x, [new_h, new_w], mode='bilinear', align_corners=False)
        return x

    def run_pretrain(self, image: Tensor, encoder_id: int = 1) -> Tensor:
        if self.heterogeneous:
            if encoder_id == 1:
                return self.dino1.get_image_embeddings(image)
            else:
                return self.dino2.get_image_embeddings(image)
        return self.dino.get_image_embeddings(image)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward1(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through encoder 1 (for heterogeneous mode)."""
        x = self._pad_to_patch_size(x)
        feats = self.dino1.get_image_embeddings(x)
        y = self.Adapter1(feats)
        return y

    def forward2(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through encoder 2 (for heterogeneous mode)."""
        x = self._pad_to_patch_size(x)
        feats = self.dino2.get_image_embeddings(x)
        y = self.Adapter2(feats)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (single encoder mode)."""
        x = self._pad_to_patch_size(x)
        feats = self.run_pretrain(x)
        y = self.Adapter(feats)
        return y

    def bi_forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Bi-temporal forward for change detection.
        
        In heterogeneous mode: x1 goes through encoder1, x2 goes through encoder2.
        In standard mode: both x1 and x2 go through the same encoder.
        """
        input_shape = x1.shape[-2:]

        if self.heterogeneous:
            y1 = self.forward1(x1)
            y2 = self.forward2(x2)
        else:
            y1 = self.forward(x1)
            y2 = self.forward(x2)

        y1_norm = y1 / (torch.norm(y1, dim=1, keepdim=True) + 1e-6)
        y2_norm = y2 / (torch.norm(y2, dim=1, keepdim=True) + 1e-6)
        sim = torch.sum(y1_norm * y2_norm, dim=1, keepdim=True)
        yc = -sim * self.logit_scale

        return F.interpolate(yc, input_shape, mode='bilinear', align_corners=False)


# Backward-compatible aliases
Dino_CD_LoRA = Dinov2_CD_LoRA
Dinov2_CD = Dinov2_CD_LoRA
