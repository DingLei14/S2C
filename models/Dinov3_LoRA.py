import os
import sys
import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils.misc import initialize_weights


working_path = os.path.abspath('.')

# Expose the local dinov3 package as a top-level import (avoid ModuleNotFoundError: dinov3)
_dinov3_root = os.path.join(working_path, 'models', 'dinov3')
if _dinov3_root not in sys.path:
    sys.path.insert(0, _dinov3_root)

# DINOv3 backbones (local, no internet)
from dinov3.hub.backbones import (
    dinov3_vitb16,
    dinov3_vitl16,
)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


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
    """Inject LoRA into SelfAttention.qkv.

    The original qkv is Linear(dim, 3*dim); only q and v channels are adapted here.
    `in_features` is exposed so DINOv3's compute_attention can read it.
    """

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
        # expose attributes expected by DINOv3 SelfAttention.compute_attention
        self.in_features = qkv.in_features
        self.dim = qkv.in_features

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)  # [B, N, 3*dim]
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # q occupies the first dim slice; v occupies the last dim slice
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class LoRA_DinoV3(nn.Module):
    def __init__(self, vit_model: nn.Module, r: int, lora_layer: Optional[List[int]] = None):
        super(LoRA_DinoV3, self).__init__()
        assert r > 0

        self.vit = vit_model
        depth = len(self.vit.blocks)
        if lora_layer is None:
            lora_layer = list(range(depth))
        self.lora_layer = lora_layer

        # store LoRA weights for checkpoint save/load
        self.w_As: List[nn.Linear] = []
        self.w_Bs: List[nn.Linear] = []

        # freeze the backbone
        for p in self.vit.parameters():
            p.requires_grad = False

        # perform qkv surgery on the selected layers
        for layer_idx, blk in enumerate(self.vit.blocks):
            if layer_idx not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)
            self.w_As.extend([w_a_q, w_a_v])
            self.w_Bs.extend([w_b_q, w_b_v])

            blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    @torch.no_grad()
    def get_image_embeddings(self, x: Tensor) -> Tensor:
        """Extract patch embeddings and reshape to [B, C, H/16, W/16]."""
        feats = self.vit.forward_features(x)
        tokens = feats["x_norm_patchtokens"]  # [B, HW, C]
        b, hw, c = tokens.shape
        h = int(round((x.shape[-2] / self.vit.patch_size)))
        w = int(round((x.shape[-1] / self.vit.patch_size)))
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return tokens


def _find_local_weights(preferred_prefixes: List[str], fallback_contains: Optional[List[str]] = None) -> Optional[str]:
    weights_dir = os.path.join(working_path, 'models', 'dinov3', 'pretrained_weights')
    if not os.path.isdir(weights_dir):
        return None
    files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
    # try preferred prefixes first
    for prefix in preferred_prefixes:
        for f in files:
            if f.startswith(prefix):
                return os.path.join(weights_dir, f)
    # fallback: match files containing keywords
    if fallback_contains:
        for key in fallback_contains:
            for f in files:
                if key in f:
                    return os.path.join(weights_dir, f)
    return None


class DinoV3_CD_LoRA(nn.Module):
    def __init__(self, num_embed: int = 16, model_name: str = 'vitl16', lora_r: int = 4, lora_layers: Optional[List[int]] = None, heterogeneous: bool = False):
        super(DinoV3_CD_LoRA, self).__init__()

        self.heterogeneous = heterogeneous
        
        # Helper function to create a ViT backbone
        def _create_vit(model_name: str):
            if model_name == 'vitl16':
                preferred = ['dinov3_vitl16_pretrain_lvd1689m', 'dinov3_vitl16_pretrain_sat493m', 'dinov3_vitl16']
                weights_path = _find_local_weights(preferred, fallback_contains=['vitl16'])
                search_dir = os.path.join(working_path, 'models', 'dinov3', 'pretrained_weights')
                print(f"[dinov3] backbone=vitl16, weights_dir={search_dir}")
                using_pretrained = bool(weights_path)
                if using_pretrained:
                    try:
                        size_mb = os.path.getsize(weights_path) / (1024*1024)
                        print(f"[dinov3] weights_path={weights_path} (size={size_mb:.2f} MB)")
                    except Exception:
                        print(f"[dinov3] weights_path={weights_path}")
                else:
                    print("[dinov3] weights_path=None (random init)")
                return dinov3_vitl16(pretrained=using_pretrained, weights=str(weights_path) if weights_path else None)
            else:
                # Default to vitb16 to align with the channel width (768) used in earlier versions
                preferred = ['dinov3_vitb16_pretrain', 'dinov3_vitb16']
                weights_path = _find_local_weights(preferred, fallback_contains=['vitb16'])
                search_dir = os.path.join(working_path, 'models', 'dinov3', 'pretrained_weights')
                print(f"[dinov3] backbone=vitb16, weights_dir={search_dir}")
                using_pretrained = bool(weights_path)
                if using_pretrained:
                    try:
                        size_mb = os.path.getsize(weights_path) / (1024*1024)
                        print(f"[dinov3] weights_path={weights_path} (size={size_mb:.2f} MB)")
                    except Exception:
                        print(f"[dinov3] weights_path={weights_path}")
                else:
                    print("[dinov3] weights_path=None (random init)")
                return dinov3_vitb16(pretrained=using_pretrained, weights=str(weights_path) if weights_path else None)
        
        # Create backbone(s)
        vit = _create_vit(model_name)
        in_channels = getattr(vit, 'embed_dim', 1024 if model_name == 'vitl16' else 768)
        self._patch_size = getattr(vit, 'patch_size', 16)
        
        if self.heterogeneous:
            # Heterogeneous mode: two separate encoders for different input modalities
            print("[dinov3] Heterogeneous mode enabled: using two separate encoders")
            vit2 = _create_vit(model_name)
            
            # Different LoRA configurations for each encoder (following Dino_LoRA_Het.py pattern)
            # Encoder 1: smaller LoRA rank, fewer layers (for one modality)
            lora_layers_1 = lora_layers if lora_layers is not None else [8, 9, 10, 11]
            self.dino1 = LoRA_DinoV3(vit, r=lora_r, lora_layer=lora_layers_1)
            
            # Encoder 2: larger LoRA rank, all layers (for another modality)
            depth = len(vit2.blocks)
            lora_layers_2 = list(range(depth))
            self.dino2 = LoRA_DinoV3(vit2, r=lora_r * 8, lora_layer=lora_layers_2)
            
            # Two separate adapters
            self.Adapter1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, num_embed, kernel_size=1),
            )
            self.Adapter2 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, num_embed, kernel_size=1),
            )
            initialize_weights(self.Adapter1, self.Adapter2)
        else:
            # Standard mode: single encoder for both inputs
            self.dino = LoRA_DinoV3(vit, r=lora_r, lora_layer=lora_layers)
            self.Adapter = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, num_embed, kernel_size=1),
            )
            initialize_weights(self.Adapter)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # expose vit attributes for external logging if needed

    @torch.no_grad()
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

    def _pad_to_patch_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad inputs to multiples of the patch size when necessary."""
        ps = self._patch_size
        if (x.shape[-2] % ps) or (x.shape[-1] % ps):
            new_h = ps * (x.shape[-2] // ps + (1 if x.shape[-2] % ps else 0))
            new_w = ps * (x.shape[-1] // ps + (1 if x.shape[-1] % ps else 0))
            x = F.interpolate(x, [new_h, new_w], mode='bilinear', align_corners=False)
        return x

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
        """Standard forward (single encoder mode only)."""
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


# Backward-compatible alias
Dinov3_CD_LoRA = DinoV3_CD_LoRA
Dino_CD_LoRA = Dinov3_CD_LoRA
Dinov3_CD = Dinov3_CD_LoRA


