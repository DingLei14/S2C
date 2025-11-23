from __future__ import annotations

import copy
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Tuple

from scipy import stats
from skimage import io
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.loss import InfoNCE, TripletLoss, loss_sparse
from utils.utils import AverageMeter, binary_accuracy as accuracy
from utils.data_parallel import BalancedDataParallel


def run_training(cfg: Dict[str, Any]) -> None:
    _prepare_directories(cfg)
    device = _get_device(cfg['training'])
    writer = SummaryWriter(str(cfg['paths']['logs']))

    dataset_module, datasets = _build_datasets(cfg)
    loaders = _build_dataloaders(datasets, cfg['training'])

    model = _build_model(cfg['model'])
    load_path = cfg['training'].get('load_path')
    if load_path:
        state_dict = torch.load(load_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained weights from: {load_path}')
    model = _wrap_model(model, cfg['training'])
    model.to(device=device)

    _sanity_check(model, device)

    best_model = _train_loop(
        loaders,
        model,
        cfg,
        writer,
        device,
        dataset_module,
    )
    writer.close()

    print('Training finished.')
    if best_model is None:
        best_model = copy.deepcopy(model)
    if best_model is not None:
        F1, acc, IoU = evaluate(loaders['test'], best_model, cfg, device, dataset_module, TTA=cfg['training'].get('tta', False))
        ckpt_dir: Path = cfg['paths']['checkpoints']
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"{cfg['experiment']['name']}_test_OA{acc * 100:.2f}_F{F1 * 100:.2f}_IoU{IoU * 100:.2f}.pth"
        torch.save(best_model.state_dict(), ckpt_dir / ckpt_name)


def _prepare_directories(cfg: Dict[str, Any]) -> None:
    for key in ('checkpoints', 'results', 'logs'):
        path: Path = cfg['paths'][key]
        path.mkdir(parents=True, exist_ok=True)


def _get_device(train_cfg: Dict[str, Any]) -> torch.device:
    if train_cfg.get('gpu', True) and torch.cuda.is_available():
        return torch.device('cuda', int(train_cfg.get('dev_id', 0)))
    return torch.device('cpu')


def _build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    module = import_module(model_cfg['module'])
    cls = getattr(module, model_cfg['class_name'])
    kwargs = model_cfg.get('args', {}) or {}
    return cls(**kwargs)


def _build_datasets(cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    dataset_cfg = cfg['dataset']
    module = import_module(dataset_cfg['module'])
    dataset_cls = getattr(module, dataset_cfg.get('class_name', 'RS'))
    _override_dataset_root(module, dataset_cfg, cfg.get('project_root'))

    datasets = {}
    for split in ('train', 'val', 'test'):
        split_cfg = dataset_cfg.get(split)
        if not split_cfg:
            raise ValueError(f'Missing dataset split config: {split}')
        kwargs = split_cfg.get('kwargs', {}) or {}
        datasets[split] = dataset_cls(split_cfg.get('split', split), **kwargs)

    return module, datasets


def _build_dataloaders(datasets: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    num_workers = int(train_cfg.get('num_workers', 4))
    pin_memory = bool(train_cfg.get('gpu', True) and torch.cuda.is_available())
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=train_cfg['train_batch_size'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=train_cfg['val_batch_size'],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=train_cfg['val_batch_size'],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        ),
    }
    return loaders


def _wrap_model(model: nn.Module, train_cfg: Dict[str, Any]) -> nn.Module:
    multi_gpu = train_cfg.get('multi_gpu')
    if not multi_gpu:
        return model

    device_ids = [int(i) for i in multi_gpu] if isinstance(multi_gpu, (list, tuple)) else [int(i) for i in str(multi_gpu).split(',')]
    if train_cfg.get('use_balanced_dp', False):
        model = BalancedDataParallel(0, model, device_ids=device_ids)
    else:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def _sanity_check(model: nn.Module, device: torch.device) -> None:
    model.eval()
    dsize = (1, 3, 512, 512)
    x1 = torch.randn(dsize, device=device)
    x2 = torch.randn(dsize, device=device)
    with torch.no_grad():
        _call_model(model, '__call__', x1)
        _call_model(model, '__call__', x2)
        _bi_forward(model, x1, x2)
    _clear_cuda_cache(device)


def _train_loop(loaders, net, cfg, writer, device, dataset_module):
    train_cfg = cfg['training']
    epochs = train_cfg['epochs']

    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    net_reserved = None

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(loaders['train']) * epochs)

    loss_triplet = TripletLoss()
    loss_infoNCE = InfoNCE(embed_dim=16)

    params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optim.SGD(params, train_cfg['lr'], weight_decay=5e-4, momentum=0.9, nesterov=True)

    enabled_losses = set(train_cfg.get('loss_terms', ['triplet', 'infoNCE', 'sparse']))
    if not enabled_losses:
        raise ValueError('`loss_terms` must contain at least one entry.')

    while curr_epoch < epochs:
        _clear_cuda_cache(device)
        net.train()
        start = time.time()
        loss_triplet_meter = AverageMeter()
        loss_infoNCE_meter = AverageMeter()
        loss_sparse_meter = AverageMeter()

        curr_iter = curr_epoch * len(loaders['train'])
        for i, data in enumerate(loaders['train']):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, train_cfg)
            imgs_A, imgs_B, imgs_A_aug, imgs_B_aug = data
            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()
            imgs_A_aug = imgs_A_aug.to(device).float()
            imgs_B_aug = imgs_B_aug.to(device).float()

            optimizer.zero_grad()
            y1 = net(imgs_A)
            y2 = net(imgs_B)
            y1_ = net(imgs_A_aug)
            y2_ = net(imgs_B_aug)
            yc = _bi_forward(net, imgs_A, imgs_B)

            loss_tri = loss_triplet(y1, y1_, y2, y2_)
            loss_cl = (loss_infoNCE(y1, y2_) + loss_infoNCE(y1_, y2)) * train_cfg['infoNCE_weight']
            loss_s = loss_sparse(yc, T=train_cfg['sparse_thred'], margin=0.0, ds_patch=16) * train_cfg['sparse_weight']

            loss = 0.0
            if 'triplet' in enabled_losses:
                loss = loss + loss_tri
            if 'infoNCE' in enabled_losses:
                loss = loss + loss_cl
            if 'sparse' in enabled_losses:
                loss = loss + loss_s
            if loss == 0:
                raise RuntimeError('Loss configuration resulted in zero total loss.')

            loss.backward()
            optimizer.step()

            loss_triplet_meter.update(loss_tri.detach().cpu().item())
            loss_infoNCE_meter.update(loss_cl.detach().cpu().item())
            loss_sparse_meter.update(loss_s.detach().cpu().item())

            curr_time = time.time() - start
            if (i + 1) % train_cfg['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %.6f] [triplet %.4f infoNCE %.4f sparse %.4f]' % (
                    curr_epoch, i + 1, len(loaders['train']), curr_time, optimizer.param_groups[0]['lr'],
                    loss_triplet_meter.val, loss_infoNCE_meter.val, loss_sparse_meter.val))
                writer.add_scalar('triplet_loss', loss_triplet_meter.val, running_iter)
                writer.add_scalar('infoNCE_loss', loss_infoNCE_meter.val, running_iter)
                writer.add_scalar('sparse_loss', loss_sparse_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, _ = validate(
            loaders['val'],
            net,
            cfg,
            device,
            dataset_module,
            writer,
            curr_epoch,
            TTA=train_cfg.get('tta', False),
        )
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            net_reserved = copy.deepcopy(net)

        print('[epoch %d/%d %.1fs] Best val acc %.2f F1 %.2f IoU %.2f' % (
            curr_epoch, epochs, time.time() - begin_time, bestacc * 100, bestF * 100, bestIoU * 100))
        curr_epoch += 1

    return net_reserved


def validate(val_loader, net, cfg, device, dataset_module, writer, curr_epoch=0, TTA=False):
    net.eval()
    _clear_cuda_cache(device)
    start = time.time()
    paths = cfg['paths']

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    Pre_meter = AverageMeter()
    Rec_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data
        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            yc = _bi_forward(net, imgs_A, imgs_B)
            yc = F.sigmoid(yc)
            if TTA:
                yc = _tta_aggregate(net, imgs_A, imgs_B, yc)
            loss = F.binary_cross_entropy_with_logits(yc, labels)
        val_loss.update(loss.detach().cpu().item())

        preds = yc.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for (pred, label) in zip(preds, labels_np):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

        if vi == 0:
            pred_color = dataset_module.Index2Color(preds[0].squeeze())
            save_path = paths['results'] / f"{cfg['experiment']['name']}_e{curr_epoch}.png"
            io.imsave(save_path, pred_color)
            print(f'Prediction saved to {save_path}')

    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f Pre %.2f Rec %.2f' % (
        curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100,
        Pre_meter.average() * 100, Rec_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', Acc_meter.average(), curr_epoch)
    writer.add_scalar('val_F1', F1_meter.average(), curr_epoch)
    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def evaluate(test_loader, net, cfg, device, dataset_module, TTA=False):
    net.eval()
    _clear_cuda_cache(device)
    start = time.time()

    acc_meter = AccMeter()
    for vi, data in enumerate(test_loader):
        imgs_A, imgs_B, labels = data
        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            yc = _bi_forward(net, imgs_A, imgs_B)
            yc = F.sigmoid(yc)
            if TTA:
                yc = _tta_aggregate(net, imgs_A, imgs_B, yc)

        preds = yc.detach().cpu().numpy() > 0.5
        labels_np = labels.detach().cpu().numpy()
        for (pred, GT) in zip(preds, labels_np):
            TP, TN, FP, FN = calc_TP(pred, GT)
            acc_meter.update(TP, TN, FP, FN)

    TP, TN, FP, FN = acc_meter.val()
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (FP + TP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN + 1e-10)
    F1 = stats.hmean([precision, recall])
    print('Test eval results: Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f.' % (
        acc * 100, precision * 100, recall * 100, F1 * 100, IoU * 100))
    return F1, acc, IoU


def _tta_aggregate(net, imgs_A, imgs_B, base_pred):
    preds = base_pred.clone()
    for dims in ([2], [3], [2, 3]):
        imgs_A_flip = torch.flip(imgs_A, dims)
        imgs_B_flip = torch.flip(imgs_B, dims)
        yc_flip = _bi_forward(net, imgs_A_flip, imgs_B_flip)
        yc_flip = torch.flip(yc_flip, dims)
        preds += F.sigmoid(yc_flip)
    preds = preds / 4.0
    return preds


def calc_TP(pred, label):
    pred = (pred >= 0.5)
    label = (label >= 0.5)
    GT = (label).sum()
    TP = (pred * label).sum()
    FP = (pred * (~label)).sum()
    FN = ((~pred) * (label)).sum()
    TN = ((~pred) * (~label)).sum()
    return TP, TN, FP, FN


class AccMeter(object):
    def __init__(self):
        self.initialized = False
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None

    def initialize(self, TP, TN, FP, FN):
        self.TP = float(TP)
        self.TN = float(TN)
        self.FP = float(FP)
        self.FN = float(FN)
        self.initialized = True

    def update(self, TP, TN, FP, FN):
        if not self.initialized:
            self.initialize(TP, TN, FP, FN)
        else:
            self.add(TP, TN, FP, FN)

    def add(self, TP, TN, FP, FN):
        self.TP += float(TP)
        self.TN += float(TN)
        self.FP += float(FP)
        self.FN += float(FN)

    def val(self):
        return self.TP, self.TN, self.FP, self.FN


def _bi_forward(model, *args, **kwargs):
    module = model.module if hasattr(model, 'module') else model
    return module.bi_forward(*args, **kwargs)


def _call_model(model, fn_name, *args, **kwargs):
    module = model.module if hasattr(model, 'module') else model
    fn = getattr(module, fn_name)
    return fn(*args, **kwargs)


def adjust_lr(optimizer, curr_iter, all_iter, train_cfg):
    scale_running_lr = ((1.0 - float(curr_iter) / all_iter) ** 1.5)
    running_lr = train_cfg['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _override_dataset_root(module, dataset_cfg: Dict[str, Any], project_root: Path | None) -> None:
    root_override = dataset_cfg.get('root')
    if not root_override:
        return
    root_path = Path(root_override)
    if not root_path.is_absolute():
        base = project_root or Path.cwd()
        root_path = base / root_path
    root_str = str(root_path)
    setter = getattr(module, 'set_root', None)
    if callable(setter):
        setter(root_str)
    elif hasattr(module, 'root'):
        setattr(module, 'root', root_str)
    else:
        raise AttributeError(
            f'Dataset module {module.__name__} does not support root override. '
            'Please implement `set_root` or expose a `root` attribute.'
        )

