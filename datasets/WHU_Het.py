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

MEAN_RGB = np.array([104.32, 121.83, 123.94])
STD_RGB = np.array([45.35, 45.29, 50.20])
MEAN_SAR = 84.78
STD_SAR = 22.69
num_classes = 1
root = '/root/autodl-fs/Data/WHU_Het/'


def set_root(new_root: str) -> None:
    """Set the dataset root directory."""
    global root
    root = new_root


def showIMG(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im, mode='rgb'):
    """Normalize image based on modality (rgb or sar)."""
    if mode == 'rgb':
        im = (im - MEAN_RGB) / STD_RGB
    else:
        im = (im - MEAN_SAR) / STD_SAR
    return im.astype(np.float32)


def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


def Color2Index(ColorLabel):
    """Convert color label to binary index map."""
    if len(ColorLabel.shape) == 3:
        IndexMap = (ColorLabel.max(axis=-1) > 0).astype(np.uint8)
    else:
        IndexMap = ColorLabel.clip(max=1)
    return IndexMap


def tensor2color(img_tensor):
    img = img_tensor.cpu().detach().numpy()
    img = exposure.rescale_intensity(img, out_range=np.uint8)
    return img


def Index2Color(pred):
    pred = pred * 255
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
        h_rate = h / c_h
        w_rate = w / c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times == 1:
            stride_h = 0
        else:
            stride_h = math.ceil(c_h * (h_times - h_rate) / (h_times - 1))
        if w_times == 1:
            stride_w = 0
        else:
            stride_w = math.ceil(c_w * (w_times - w_rate) / (w_times - 1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j * c_h - j * stride_h)
                if j == (h_times - 1):
                    s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i * c_w - i * stride_w)
                if i == (w_times - 1):
                    s_w = w - c_w
                e_w = s_w + c_w
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                if label_dims == 2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' % len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels


def rand_crop_CD(img1, img2, label, size):
    h = img1.shape[0]
    w = img1.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
        return img1, img2, label
    else:
        s_h = random.randint(0, h - c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w - c_w)
        e_w = s_w + c_w

        crop_im1 = img1[s_h:e_h, s_w:e_w, :]
        crop_im2 = img2[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        return crop_im1, crop_im2, crop_label


def rand_flip_CD(img1, img2, label):
    r = random.random()
    if r < 0.25:
        return img1, img2, label
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()


def read_RSimages(mode, read_list=False):
    """Read heterogeneous remote sensing images (RGB + SAR)."""
    assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'rgb')
    img_B_dir = os.path.join(root, mode, 'sar')
    label_dir = os.path.join(root, mode, 'mask')

    data_list = os.listdir(img_A_dir)
    data_A, data_B, labels = [], [], []
    for idx, it in enumerate(data_list):
        if it[-4:] == '.png' or it[-4:] == '.tif':
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it[:-4] + '.png')
            label_path = os.path.join(label_dir, it[:-4] + '.png')

            img_A = io.imread(img_A_path)
            img_B = io.imread(img_B_path)
            # SAR is single channel, expand to 3 channels
            if len(img_B.shape) == 2:
                img_B = np.repeat(np.expand_dims(img_B, 2), 3, 2)
            label = io.imread(label_path)

            # Normalize with modality-specific statistics
            img_A = normalize_image(img_A, 'rgb')
            img_B = normalize_image(img_B, 'sar')
            data_A.append(img_A)
            data_B.append(img_B)
            labels.append(Color2Index(label))
        if not idx % 50:
            print('%d/%d images loaded.' % (idx, len(data_list)))
    print(data_A[0].shape)
    print(str(len(data_A)) + ' ' + mode + ' images loaded.')
    return data_A, data_B, labels


def weak_aug(img1, img2, mask):
    """Weak augmentation for both images and mask."""
    h, w, _ = img1.shape
    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], p=1., additional_targets={'image2': 'image', 'mask': 'mask'})

    tf_sample = aug(image=img1, image2=img2, mask=mask)
    return tf_sample['image'], tf_sample['image2'], tf_sample['mask']


def strong_aug_rgb(img, img_ref):
    """Strong augmentation for RGB images."""
    aug = A.Compose([
        # Color transform
        A.PixelDistributionAdaptation(reference_images=[img_ref], read_fn=lambda x: x, p=0.5),
        A.RGBShift(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        # Blur and noise
        A.Downscale(scale_range=[0.25, 0.5], p=0.5),
        # Spatial transform
        A.ShiftScaleRotate(shift_limit=0.03125, scale_limit=0., rotate_limit=0., p=1.),
    ], p=1.)
    aug_result = aug(image=img)
    return aug_result['image']


def strong_aug_sar(img, img_ref):
    """Strong augmentation for SAR images (no color transform)."""
    aug = A.Compose([
        # Brightness/contrast only
        A.RandomBrightnessContrast(p=0.5),
        # Blur and noise
        A.Downscale(scale_range=[0.25, 0.5], p=0.5),
        # Spatial transform
        A.ShiftScaleRotate(shift_limit=0.03125, scale_limit=0., rotate_limit=0., p=1.),
    ], p=1.)
    aug_result = aug(image=img)
    return aug_result['image']


class RS(data.Dataset):
    """WHU Heterogeneous Change Detection Dataset (RGB + SAR).
    
    This dataset contains heterogeneous image pairs:
    - Image A: RGB optical images
    - Image B: SAR (Synthetic Aperture Radar) images
    
    Args:
        mode: 'train', 'val', or 'test'
        random_crop: whether to apply random crop
        crop_nums: number of crops per image when random_crop is True
        sliding_crop: whether to apply sliding window crop
        crop_size: size of the crop window
        random_flip: whether to apply random flip
        epoch_sample_size: number of image pairs to sample per epoch for training
    """

    def __init__(self, mode, random_crop=False, crop_nums=6, sliding_crop=False, 
                 crop_size=512, random_flip=False, epoch_sample_size=None):
        self.mode = mode
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        self.epoch_sample_size = epoch_sample_size

        data_A, data_B, labels = read_RSimages(mode, read_list=False)
        if sliding_crop:
            data_A, data_B, labels = sliding_crop_CD(data_A, data_B, labels, [self.crop_size, self.crop_size])
        self.data_A, self.data_B, self.labels = data_A, data_B, labels
        self.total_samples = len(self.data_A)

        # For training mode, use epoch sampling if specified
        if self.mode == 'train' and self.epoch_sample_size is not None and self.epoch_sample_size > 0:
            self.use_epoch_sampling = True
            self.epoch_sample_size = min(self.epoch_sample_size, self.total_samples)
            self.sample_indices = []
            self.shuffle_epoch()
        else:
            self.use_epoch_sampling = False
            self.sample_indices = list(range(self.total_samples))

        self._update_len()

    def shuffle_epoch(self):
        """Shuffle and sample indices for a new epoch."""
        if self.use_epoch_sampling:
            all_indices = list(range(self.total_samples))
            random.shuffle(all_indices)
            self.sample_indices = all_indices[:self.epoch_sample_size]
            print(f'Epoch sampling: {self.epoch_sample_size}/{self.total_samples} pairs selected.')

    def _update_len(self):
        """Update dataset length based on current sample indices."""
        base_len = len(self.sample_indices)
        if self.random_crop:
            self.len = self.crop_nums * base_len
        else:
            self.len = base_len

    def __getitem__(self, idx):
        if self.random_crop:
            sample_idx = self.sample_indices[idx // self.crop_nums]
        else:
            sample_idx = self.sample_indices[idx]

        data_A = self.data_A[sample_idx].copy()
        data_B = self.data_B[sample_idx].copy()
        label = self.labels[sample_idx].copy()

        if self.random_crop:
            data_A, data_B, label = rand_crop_CD(data_A, data_B, label, [self.crop_size, self.crop_size])

        if self.mode == 'train':
            data_A, data_B, label = weak_aug(data_A, data_B, label)
            # Use modality-specific strong augmentation
            data_A_aug = strong_aug_rgb(data_A, data_B)
            data_B_aug = strong_aug_sar(data_B, data_A)
            return F.to_tensor(data_A), F.to_tensor(data_B), F.to_tensor(data_A_aug), F.to_tensor(data_B_aug)
        else:
            return F.to_tensor(data_A), F.to_tensor(data_B), label

    def __len__(self):
        return self.len
