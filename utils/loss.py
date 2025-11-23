import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from math import pi
# Recommend

class CosineEmbeddingLoss(nn.Module):
    def __init__(self, reduction='mean', T=3.0):
        super(CosineEmbeddingLoss, self).__init__()
        self.T = T
        self.neg_weight = 1.
        
    def forward(self, x1, x2, yc):
        b,c,h,w = x1.size()
                
        #x1_norm = F.softmax(x1/self.T, dim=1)
        #x2_norm = F.softmax(x2/self.T, dim=1)
        x1_norm = x1 / torch.norm(x1, dim=1, keepdim=True) #F.softmax(feat1, dim=1)
        x2_norm = x2 / torch.norm(x2, dim=1, keepdim=True) #F.softmax(feat2, dim=1)
        y_c = F.sigmoid(yc)
        y_nc = 1-yc
                
        cos = torch.cosine_similarity(x1_norm, x2_norm, dim=1)
                
        change_loss = cos*y_c
        unchange_loss = (1-cos)*y_nc
                        
        return (change_loss + unchange_loss).mean()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class InfoNCE_naive(nn.Module):
    def __init__(self, embed_dim=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        self.device = device

    def forward(self, featA, featB):
    
        B, C, _, _ = featA.shape
        
        featA = self.pool(featA).squeeze()
        featB = self.pool(featB).squeeze()
        featA = F.normalize(featA, dim=1)
        featB = F.normalize(featB, dim=1)
                        
        logits_A = featA @ featB.transpose(0,1)   #[B,C] * [C,B] -> [B,B]
        logits_A *= self.logit_scale
        logits_B = logits_A.T
        
        labels = torch.arange(len(logits_A), dtype=torch.long, device=self.device)    
        loss = self.loss_function(logits_A, labels) + self.loss_function(logits_B, labels)
        return loss

class InfoNCE(nn.Module):
    def __init__(self, embed_dim=16, patch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
        #self.proj = nn.Sequential(torch.nn.AdaptiveAvgPool2d([4,4]), ConvBatchNormReLU(embed_dim, embed_dim, 1, 1, 0, 1, leaky=True)).to(device)
        self.proj = torch.nn.AdaptiveAvgPool2d([patch_size, patch_size])
        #self.proj = torch.nn.AdaptiveMaxPool2d([patch_size, patch_size])
        self.pool = torch.nn.AdaptiveAvgPool1d(1).to(device)
        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        self.device = device

    def forward(self, featA, featB):
        B, C, _, _ = featA.shape
    
        featA = self.proj(featA)
        featB = self.proj(featB)
        
        norm_featA = featA / torch.norm(featA, dim=1, keepdim=True)
        norm_featB = featB / torch.norm(featB, dim=1, keepdim=True)     
        
        norm_featA = norm_featA.flatten(2) #[B,C,G]
        norm_featB = norm_featB.flatten(2) #[B,C,G]
        
        norm_featA = norm_featA.permute(2, 0, 1) #[B,C,G]->[G,B,C]
        norm_featB = norm_featB.permute(2, 1, 0) #[B,C,G]->[G,C,B]
        
        logits_A = norm_featA @ norm_featB  #[G,B,C] * [G,C,B] -> [G,B,B]
        logits_A = logits_A.permute(1,2,0)  #[G,B,B] -> [B,B,G]
        logits_A =  self.pool(logits_A).squeeze() #*self.logit_scale
        logits_B = logits_A.T
        
        labels = torch.arange(len(logits_A), dtype=torch.long, device=self.device)    
        loss = self.loss_function(logits_A, labels) + self.loss_function(logits_B, labels)
        return loss
        
def loss_sparse(yc, T=0.2, margin=0.1, ds_patch=8):
    y_c = F.sigmoid(yc)
    avg_sparse_loss = F.relu(y_c.mean()-T)
    
    b, c, h, w = yc.shape
    patch_num = int(b*h*w/(ds_patch*ds_patch))
    thred_num = int(patch_num*(1-T))
    
    y_c = F.interpolate(y_c, scale_factor=1/ds_patch, mode='bilinear')
    y_c = y_c.permute(1,0,2,3).contiguous().view(c,-1)
    
    values, indices = torch.topk(y_c, thred_num, dim=-1, largest=False)
    grid_sparse_loss = torch.mean(F.relu(values-margin))
    #grid_sparse_loss = torch.mean(1-torch.cos(y_c*pi))
    
    return grid_sparse_loss + avg_sparse_loss

class TripletLossBCE_mask(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TripletLossBCE_mask, self).__init__()
        self.device = device
        self.margin = 0.1
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1).to(device)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1).to(device)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
        self.loss_function = torch.nn.CrossEntropyLoss().to(device)

    def forward(self, featA, featA_aug, featB, featB_aug):
        loss_1 = self.single_forward(featA, featA_aug, featB)
        loss_2 = self.single_forward(featB, featB_aug, featA)
        return loss_1+loss_2

    def single_forward(self, feat1, feat1_aug, feat2):  
        n = feat1.size(0)
        labels = torch.ones([n], dtype=torch.long, device=self.device)
                      
        norm_feat1 = feat1 / torch.norm(feat1, dim=1, keepdim=True) #F.softmax(feat1, dim=1)
        norm_feat2 = feat2 / torch.norm(feat2, dim=1, keepdim=True) #F.softmax(feat2, dim=1)
        norm_feat1_aug = feat1_aug / torch.norm(feat1_aug, dim=1, keepdim=True) #F.softmax(feat1_aug, dim=1)
        
        # Compute similarity matrix
        sim_11 = torch.sum(norm_feat1*norm_feat1_aug, dim=1).view([n,-1])
        sim_12 = torch.sum(norm_feat1*norm_feat2, dim=1).view([n,-1])
        #sim_12_aug = torch.sum(norm_feat1_aug*norm_feat2, dim=1).view([n,-1])
        y_nc = F.sigmoid(sim_12*self.logit_scale)
        y_c = 1-y_nc
        
        sim_logits_11 = sim_11*self.logit_scale
        sim_logits_12 = sim_12*self.logit_scale
        sim_logits = torch.cat([sim_logits_12, sim_logits_11], dim=1)
        
        dif_logits_11 = -sim_11*self.logit_scale
        dif_logits_12 = -sim_12*self.logit_scale
        dif_logits = torch.cat([dif_logits_11, dif_logits_12], dim=1)
                
        loss_sim_mask = self.loss_function(sim_logits, labels)*y_nc
        loss_dif_mask = self.loss_function(dif_logits, labels)*y_c
        
        loss = (loss_sim_mask + loss_dif_mask).mean()
        return loss

class TripletLossBCE(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TripletLossBCE, self).__init__()
        self.device = device
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1).to(device)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1).to(device)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
        self.loss_function = torch.nn.CrossEntropyLoss().to(device)

    def forward(self, featA, featA_aug, featB, featB_aug):
        loss_1 = self.single_forward(featA, featA_aug, featB)
        loss_2 = self.single_forward(featB, featB_aug, featA)
        return loss_1+loss_2

    def single_forward(self, feat1, feat1_aug, feat2):  
        n = feat1.size(0)
                      
        norm_feat1 = feat1 / torch.norm(feat1, dim=1, keepdim=True)
        norm_feat2 = feat2 / torch.norm(feat2, dim=1, keepdim=True)       
        norm_feat1_aug = feat1_aug / torch.norm(feat1_aug, dim=1, keepdim=True) 
        
        
        # Compute similarity matrix
        sim_11 = torch.sum(norm_feat1*norm_feat1_aug, dim=1).view([n,-1])
        sim_12 = torch.sum(norm_feat1*norm_feat2, dim=1).view([n,-1])
                
        labels = torch.ones([n], dtype=torch.long, device=self.device)
        
        sim_logits_11 = self.avgpool(sim_11)*self.logit_scale
        sim_logits_12 = self.avgpool(sim_12)*self.logit_scale
        sim_logits = torch.cat([sim_logits_12, sim_logits_11], dim=1)
        sim_loss = self.loss_function(sim_logits, labels)
        
        dif_logits_11 = self.avgpool(-sim_11)*self.logit_scale
        dif_logits_12 = self.avgpool(-sim_12)*self.logit_scale   
        dif_logits = torch.cat([dif_logits_11, dif_logits_12], dim=1)
        dif_loss = self.loss_function(dif_logits, labels)
        
        loss = sim_loss + dif_loss
        return loss

class TripletLoss_HT(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TripletLoss_HT, self).__init__()
        #self.loss_triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)        
        self.loss_triplet = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: F.cosine_similarity(x, y))
        #self.loss_ranking = nn.MarginRankingLoss()

    def forward(self, featA, featA_aug, featB):        
        loss = self.loss_triplet(featA, featB, featA_aug)
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TripletLoss, self).__init__()
        #self.loss_triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)        
        self.loss_triplet = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
        #self.loss_ranking = nn.MarginRankingLoss()

    def forward(self, featA, featA_aug, featB, featB_aug):
        loss_1 = self.single_forward(featA, featA_aug, featB)
        loss_2 = self.single_forward(featB, featB_aug, featA)
        return loss_1+loss_2

    def single_forward(self, feat1, feat1_aug, feat2):      
        n = feat1.size(0)
        
        loss = self.loss_triplet(feat1, feat1_aug, feat2)
        
        return loss.mean()
        
# github.com/Jeff-Zilence/TransGeo2022/blob/main/criterion/soft_triplet.py
class SoftTripletBiLoss(nn.Module):
    def __init__(self, alpha=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SoftTripletBiLoss, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1).to(device)
        self.alpha = alpha

    def forward(self, inputs_q, inputs_k):
        loss_1, mean_pos_sim_1, mean_neg_sim_1 = self.single_forward(inputs_q, inputs_k)
        loss_2, mean_pos_sim_2, mean_neg_sim_2 = self.single_forward(inputs_k, inputs_q)
        return (loss_1+loss_2)*0.5 #, (mean_pos_sim_1+mean_pos_sim_2)*0.5, (mean_neg_sim_1+mean_neg_sim_2)*0.5

    def single_forward(self, featA, featB):
        n = featB.size(0)
        
        inputs_q = self.pool(featA).squeeze()
        inputs_k = self.pool(featB).squeeze()
        
        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())

        # split the positive and negative pairs
        eyes_ = torch.eye(n).cuda()

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        pos_sim_ = pos_sim.unsqueeze(dim=1).expand(n, n-1)
        neg_sim_ = neg_sim.reshape(n, n-1)

        loss_batch = torch.log(1 + torch.exp((neg_sim_ - pos_sim_) * self.alpha))
        if torch.isnan(loss_batch).any():
            print(inputs_q, inputs_k)
            raise Exception

        loss = loss_batch.mean()
        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()
        return loss, mean_pos_sim, mean_neg_sim

# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
    
def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)
    
    if weight_pos is not None:        
        loss = weight_pos * (target * torch.log(output)) + \
               weight_neg * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.2, weight_neg=0.8):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class LatentSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean', T=1.0):
        super(LatentSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        self.T = T
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1/self.T, dim=1)
        x2 = F.softmax(x2/self.T, dim=1)
        
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float() - label_change.float()
        target = torch.reshape(target, [b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss
        
class ChangeSalience(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.MSELoss(reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)[:,0,:,:]
        x2 = F.softmax(x2, dim=1)[:,0,:,:]
                
        loss = self.loss_f(x1, x2.detach()) + self.loss_f(x2, x1.detach())
        return loss*0.5
    

def pix_loss(output, target, pix_weight, ignore_index=None):
    # Calculate log probabilities
    if ignore_index is not None:
        active_pos = 1-(target==ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos
        
    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))
    # Multiply with weights
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss
    

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
        instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        if instance:
            self.add_module(
                "bn",
                nn.InstanceNorm2d(num_features=out_channels),
            )
        else:
            self.add_module(
                "bn",
                nn.BatchNorm2d(
                    num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
                ),
            )

        if leaky:
            self.add_module("relu", nn.LeakyReLU(0.1))
        elif relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(ConvBatchNormReLU, self).forward(x)

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]