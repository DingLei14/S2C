
import math
import os
import sys
import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import Dict, List
from safetensors import safe_open
from safetensors.torch import save_file
from utils.misc import initialize_weights

working_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_sam2_root = os.path.join(working_path, 'models', 'sam2')
if _sam2_root not in sys.path:
    sys.path.insert(0, _sam2_root)

from models.sam2.sam2.build_sam import build_sam2


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

def build_efficient_sam_vitt():    
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint = working_path+'/EfficientSAM/weights/efficient_sam_vitt.pt',
    ).eval()


def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint = working_path+'/EfficientSAM/weights/efficient_sam_vits.pt',
    ).eval()

class _LoRA_qkv(nn.Module):
    """LoRA wrapper for qkv linear layer.
    
    SAM2 uses MultiScaleAttention where:
    self.qkv = nn.Linear(dim, dim_out * 3)
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
    q, k, v = torch.unbind(qkv, 2)
    
    Note: dim_out may differ from dim in SAM2's Hiera backbone.
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
        # SAM2: qkv output is dim_out * 3, so each q/k/v has dimension out_features // 3
        self.dim_out = qkv.out_features // 3

    def forward(self, x):
        qkv = self.qkv(x)  # B, H, W, 3*dim_out
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # q is in the first dim_out, v is in the last dim_out
        qkv[..., :self.dim_out] += new_q
        qkv[..., -self.dim_out:] += new_v
        return qkv

class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, r: int, sam_model: nn.Module=None, lora_layer=[8,9,10,11]): #lora_layer=None
        super(LoRA_Sam, self).__init__()

        if not sam_model:
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            checkpoint = os.path.join(working_path, "models", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
            sam_model = build_sam2(model_cfg, checkpoint)
            #sam_model = build_efficient_sam_vits()
        
        assert r > 0
        
        # SAM2 image_encoder structure: image_encoder.trunk.blocks (Hiera backbone)
        trunk = sam_model.image_encoder.trunk
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(trunk.blocks)))
        
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery on trunk.blocks
        for t_layer_i, blk in enumerate(trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            dim_out = w_qkv_linear.out_features // 3
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim_out, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim_out, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,)
        self.reset_parameters()
        self.sam = sam_model
        #print(self.sam)
    
    def get_image_embeddings(self, x: Tensor) -> Tensor:
        # SAM2 image_encoder returns a dict: {"vision_features": src, "vision_pos_enc": pos, "backbone_fpn": features}
        output = self.sam.image_encoder(x)
        return output["vision_features"]

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.
        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.
        pip install safetensor if you do not have one installed yet.        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.
        pip install safetensor if you do not have one installed yet.\
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class SAM_LoRA(nn.Module):
    def __init__(self, num_embed=16):
        super(SAM_LoRA, self).__init__()
        self.sam = LoRA_Sam(r=4)
        
        self.Adapter = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, num_embed, kernel_size=1))
                                     
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #self.out_conv = nn.Conv2d(1, 1, 1)
                                        
        initialize_weights(self.Adapter)
        #self.out_conv.weight.data.fill_(-4.4162)
        #self.out_conv.bias.data.fill_(-1.3795)

    def run_pretrain(self, image):
        image_embeddings = self.sam.get_image_embeddings(image)
        return image_embeddings

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        feats = self.run_pretrain(x)        
        y = self.Adapter(feats)        
        return y

    def bi_forward(self, x1: torch.Tensor, x2: torch.Tensor):
        input_shape = x1.shape[-2:]
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        
        y1_norm = y1 / torch.norm(y1, dim=1, keepdim=True)
        y2_norm = y2 / torch.norm(y2, dim=1, keepdim=True)
        sim = torch.sum(y1_norm*y2_norm, dim=1, keepdim=True)                       
        yc = -sim*self.logit_scale
        #yc = self.out_conv(yc)
        
        return F.interpolate(yc, input_shape, mode="bilinear", align_corners=True)