import time
import os
import copy
from scipy import stats
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

working_path = os.path.abspath('.')

from utils.loss import InfoNCE, TripletLoss_HT, loss_sparse
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter

###################### Data and Model ########################
from models.effSAM_Res_Het import effSAM_Res_Het as Net
NET_NAME = 'effSAM_Res_Het'

#from datasets import SECOND_aug as RS
#DATA_NAME = 'SECOND'
from datasets import WHU_Het as RS
DATA_NAME = 'WHU_Het'
###################### Data and Model ########################


######################## Parameters ########################
args = {
    'train_batch_size': 12,
    'val_batch_size': 12,
    'lr': 0.01,
    'epochs': 20,
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None, #"0,1"
    'print_freq': 10,
    'predict_step': 5,
    'triplet_weight': 1.,
    'infoNCE_weight': 1.,
    'sparse_weight': 1,
    'sparse_thred': 0.2,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': '/root/tf-logs',
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'effSAM_CD_LoRA_sim_e25_OA97.03_F67.73_IoU55.50.pth')}
###################### Data and Model ######################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])

def main():
    net = Net()
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
    if args['multi_gpu']:
        net = torch.nn.DataParallel(net, [int(id) for id in args['multi_gpu'].split(',')])
    net.to(device=torch.device('cuda', int(args['dev_id'])))
    
    # test run
    dsize = (1, 3, 256, 256)
    x1 = torch.randn(dsize).cuda()
    dsize = (1, 3, 256, 256)
    x2 = torch.randn(dsize).cuda()
    net(x1, x2)
    torch.cuda.empty_cache()
    
    train_set = RS.RS('train', random_crop=False, crop_size=512) #'5_train_supervised', 
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.RS('val', sliding_crop=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    test_set = RS.RS('test', sliding_crop=False)
    test_loader = DataLoader(test_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    train(train_loader, net, val_loader, test_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, val_loader, test_loader):
    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 10.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    
    loss_triplet = TripletLoss_HT()
    loss_infoNCE = InfoNCE(embed_dim=16) #InfoNCE(embed_dim=16)
    
    params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optim.SGD(params, args['lr'], weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        loss_triplet_meter = AverageMeter()
        loss_infoNCE_meter = AverageMeter()
        loss_sparse_meter = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, imgs_A_aug = data
            if args['gpu']:
                imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
                imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
                imgs_A_aug = imgs_A_aug.to(torch.device('cuda', int(args['dev_id']))).float()
            
            optimizer.zero_grad()
            y1 = net.single_forward(imgs_A, 1)
            y2 = net.single_forward(imgs_B, 2)
            y1_ = net.single_forward(imgs_A_aug, 1)
            yc = net(imgs_A, imgs_B)
            
            loss_tri = loss_triplet(y1, y1_, y2) *args['triplet_weight']
            loss_cl = loss_infoNCE(y1, y2) *args['infoNCE_weight']
            #loss_cl = ( loss_infoNCE(y1, y2) + loss_infoNCE(y1_, y2) )*args['infoNCE_weight']
            loss_s = loss_sparse(yc, T=args['sparse_thred'], margin=0., ds_patch=16) *args['sparse_weight']
            
            loss = loss_tri+loss_cl +loss_s
            loss.backward()
            optimizer.step()

            loss_triplet_meter.update(loss_tri.cpu().detach().numpy())
            loss_infoNCE_meter.update(loss_cl.cpu().detach().numpy())
            loss_sparse_meter.update(loss_s.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [triplet loss %.4f infoNCE loss %.4f sparse loss %.4f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],\
                    loss_triplet_meter.val, loss_infoNCE_meter.val, loss_sparse_meter.val))
                writer.add_scalar('triplet_loss', loss_triplet_meter.val, running_iter)
                writer.add_scalar('infoNCE_loss', loss_infoNCE_meter.val, running_iter)
                writer.add_scalar('sparse_loss', loss_sparse_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, val_loss = validate(val_loader, net, curr_epoch)
        #evaluate(test_loader, net, TTA=True)
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            bestloss = val_loss
            net_reserved = copy.deepcopy(net)
            #torch.save(stat_dict_reserved, os.path.join(args['chkpt_dir'], NET_NAME + 'e%d_OA%.2f_F%.2f_IoU%.2f.pth' % (curr_epoch, val_acc * 100, val_F * 100, val_IoU * 100)))
        print('[epoch %d/%d %.1fs] Best val rec: %.2f, F1 score: %.2f IoU %.2f' \
              % (curr_epoch, args['epochs'], time.time() - begin_time, bestacc * 100, bestF * 100, bestIoU * 100))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            F, acc, IoU = evaluate(test_loader, net_reserved, TTA=True)
            print('Test acc: OA %.2f F %.2f IoU %.2f.' % (acc * 100, F * 100, IoU * 100))
            torch.save(net_reserved, os.path.join(args['chkpt_dir'], NET_NAME + '_test_OA%.2f_F%.2f_IoU%.2f.pth' % (acc * 100, F * 100, IoU * 100)))
            return

def validate(val_loader, net, curr_epoch=0, TTA=False):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    Pre_meter = AverageMeter()
    Rec_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data
        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            yc = net(imgs_A, imgs_B)
            yc = F.sigmoid(yc)
            if TTA:
                imgs_A_v = torch.flip(imgs_A, [2])
                imgs_B_v = torch.flip(imgs_B, [2])
                yc_v = net(imgs_A_v, imgs_B_v)
                yc_v = torch.flip(yc_v, [2])
                yc += F.sigmoid(yc_v)
                
                imgs_A_h = torch.flip(imgs_A, [3])
                imgs_B_h = torch.flip(imgs_B, [3])
                yc_h = net(imgs_A_h, imgs_B_h)
                yc_h = torch.flip(yc_h, [3])
                yc += F.sigmoid(yc_h)
                
                imgs_A_hv = torch.flip(imgs_A, [2,3])
                imgs_B_hv = torch.flip(imgs_B, [2,3])
                yc_hv = net(imgs_A_hv, imgs_B_hv)
                yc_hv = torch.flip(yc_hv, [2,3])
                yc += F.sigmoid(yc_hv)                    
                yc = yc/4.0            
            loss = F.binary_cross_entropy_with_logits(yc, labels)
        val_loss.update(loss.cpu().detach().numpy())

        preds = yc.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

        if not vi:
            #if not curr_epoch:
                #img_A = RS.tensor2color(imgs_A[0])
                #img_B = RS.tensor2color(imgs_B[0])
                #GT_color = RS.Index2Color(labels[0].squeeze())
                #io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_imgA.png'), img_A)
                #io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_imgB.png'), img_B)
                #io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_GT.png'), GT_color)
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_e%d'%curr_epoch+'.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f Pre %.2f Rec %.2f' % (
    curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100,
    Pre_meter.average() * 100, Rec_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', Acc_meter.average(), curr_epoch)

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg

def calc_TP(pred, label):
    pred = (pred>= 0.5)
    label = (label>= 0.5)
    GT = (label).sum()
    TP = (pred * label).sum()
    FP = (pred * (~label)).sum()
    FN = ((~pred) * (label)).sum()
    TN = ((~pred) * (~label)).sum()
    return TP, TN, FP, FN

class AccMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.count = None

    def initialize(self, TP, TN, FP, FN):    
        self.TP = float(TP)
        self.TN = float(TN)
        self.FP = float(FP)
        self.FN = float(FN)
        self.count = 1
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
        self.count += 1        
        
    def val(self):
        return self.TP, self.TN, self.FP, self.FN

def evaluate(test_loader, net, TTA=False):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    acc_meter = AccMeter()
    for vi, data in enumerate(test_loader):
        imgs_A, imgs_B, labels = data
        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            yc = net(imgs_A, imgs_B)
            yc = F.sigmoid(yc)
            if TTA:
                imgs_A_v = torch.flip(imgs_A, [2])
                imgs_B_v = torch.flip(imgs_B, [2])
                yc_v = net(imgs_A_v, imgs_B_v)
                yc_v = torch.flip(yc_v, [2])
                yc += F.sigmoid(yc_v)
                
                imgs_A_h = torch.flip(imgs_A, [3])
                imgs_B_h = torch.flip(imgs_B, [3])
                yc_h = net(imgs_A_h, imgs_B_h)
                yc_h = torch.flip(yc_h, [3])
                yc += F.sigmoid(yc_h)
                
                imgs_A_hv = torch.flip(imgs_A, [2,3])
                imgs_B_hv = torch.flip(imgs_B, [2,3])
                yc_hv = net(imgs_A_hv, imgs_B_hv)
                yc_hv = torch.flip(yc_hv, [2,3])
                yc += F.sigmoid(yc_hv)                    
                yc = yc/4.0

        preds = yc.cpu().detach().numpy()>0.5
        labels = labels.cpu().detach().numpy()
        for (pred, GT) in zip(preds, labels):
            TP, TN, FP, FN = calc_TP(pred, GT)
            acc_meter.update(TP, TN, FP, FN)
            
    TP, TN, FP, FN = acc_meter.val()
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)
    IoU = TP / (FP+TP+FN+1e-10)
    acc = (TP+TN) / (TP+FP+FN+TN+1e-10)
    F1 = stats.hmean([precision, recall])
    print('Test eval results: Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f.'%(acc*100, precision*100, recall*100, F1*100, IoU*100))

    return F1, acc, IoU


def defreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train() 

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()        

def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 1.5)
    running_lr = args['lr'] * scale_running_lr + 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
