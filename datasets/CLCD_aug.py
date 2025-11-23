import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from skimage.transform import rescale
from torchvision.transforms import functional as F
import warnings

import albumentations as A 

warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform",
    module="albumentations",
)

num_classes = 1
root = 'data/CLCD'

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    #im = (im - MEAN) / STD
    im = im/255
    return im.astype(np.float32)

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap

def tensor2color(img_tensor):
    img = img_tensor.cpu().detach().numpy()
    img = exposure.rescale_intensity(img, out_range=np.uint8)
    return img

def Index2Color(pred):
    #pred = exposure.rescale_intensity(pred, out_range=np.uint8)
    pred = pred*255
    return pred.astype(np.uint8)

def sliding_crop_CD(imgs1, imgs2, labels, size):
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img1, img2, label in zip(imgs1, imgs2, labels):
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_labels.append(label)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' %len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels

def rand_crop_CD(img1, img2, label, size):
    # print(img.shape)
    h = img1.shape[0]
    w = img1.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im1 = img1[s_h:e_h, s_w:e_w, :]
        crop_im2 = img2[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im1, crop_im2, crop_label

def rand_flip_CD(img1, img2, label):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img1, img2, label
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()

def read_RSimages(mode, read_list=False):
    #assert mode in ['train0', 'val0', 'test0']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_dir = os.path.join(root, mode, 'label')
    
    data_list = os.listdir(img_A_dir)
    #if mode=='train': random.shuffle(data_list)
        
    data_A, data_B, labels = [], [], []
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_path = os.path.join(label_dir, it)
            
            img_A = io.imread(img_A_path)
            img_B = io.imread(img_B_path)
            label = io.imread(label_path)
            
            data_A.append(img_A)
            data_B.append(img_B)
            labels.append(Color2Index(label))
        #if mode=='train' and len(data_A)>99: break
        if not idx%10: print('%d/%d images loaded.'%(idx, len(data_list)))
    print(data_A[0].shape)
    print(str(len(data_A)+1) + ' ' + mode + ' images loaded.')   
    return data_A, data_B, labels

def weak_aug(img1, img2, mask):
    h, w, _ = img1.shape
    aug = A.Compose([
        A.RandomResizedCrop(size=(h,w),scale=(0.75, 1.0), p=0.5),
        A.SquareSymmetry(p=1.0),
    ], p=1., additional_targets={'image2':'image', 'mask':'mask'})
     
    tf_sample = aug(image=img1, image2=img2, mask=mask)
    return tf_sample['image'], tf_sample['image2'], tf_sample['mask']

def strong_aug(img, img_ref):
    aug = A.Compose([
        ################ color transform ###############
        A.PixelDistributionAdaptation(reference_images=[img_ref], read_fn=lambda x: x, p=1.),
        A.RGBShift(p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        ################# blur and noise ###############
        #A.Blur(p=0.5),
        A.Downscale(scale_range=[0.25, 0.5], p=0.5), #interpolation={"downscale":cv2.INTER_NEAREST, "upscale":cv2.INTER_LINEAR}, 
        #A.ISONoise(p=0.5),
        ############### spatial transform ##############
        #A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.0, rotate_limit=0.0, p=1.),
        ], p=1.)
    aug_result = aug(image=img)
    return aug_result['image']

class RS(data.Dataset):
    def __init__(self, mode, random_crop=False, crop_nums=6, sliding_crop=False, crop_size=512, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        data_A, data_B, labels = read_RSimages(mode, read_list=False)
        if sliding_crop:
            data_A, data_B, labels = sliding_crop_CD(data_A, data_B, labels, [self.crop_size, self.crop_size])                
        self.data_A, self.data_B, self.labels = data_A, data_B, labels
        if self.random_crop:
            self.len = crop_nums*len(self.data_A)
        else:
            self.len = len(self.data_A)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = idx//self.crop_nums
        data_A = self.data_A[idx]
        data_B = self.data_B[idx]
        label = self.labels[idx]
                
        #if self.random_crop:
        #    data_A, data_B, label = rand_crop_CD(data_A, data_B, label, [self.crop_size, self.crop_size])
        if self.mode=='train':        
           data_A, data_B, label = weak_aug(data_A, data_B, label)
           data_A_aug = strong_aug(data_A, data_B)
           data_B_aug = strong_aug(data_B, data_A)
           return F.to_tensor(data_A), F.to_tensor(data_B), F.to_tensor(data_A_aug), F.to_tensor(data_B_aug) #, label
        else:
            return F.to_tensor(data_A), F.to_tensor(data_B), label

    def __len__(self):
        return self.len
