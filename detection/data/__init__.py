import random

import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa

class cnormalize(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_bgr (bool): Whether to convert the image from RGB to BGR,
            default is true.
    """

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1), to_01=False, to_bgr=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_01 = to_01
        self.to_bgr = to_bgr

    def __call__(self, img):
        img = img.astype(np.float32)
        if self.to_01:
            img = img / 255.0
        if self.to_bgr:
            img = img[:, :, [2, 1, 0]]
        img = (img - self.mean) / self.std
        return img

def convert_style(img, domain_style=0):
    ih, iw, ic = img.shape
    img = img.reshape(1, ih, iw, ic)
    if domain_style == 0:   ##origin
        return img.reshape(ih, iw, ic)
    elif domain_style == 1: ##color and contrast style
        aug_bright = iaa.AddToBrightness((-30,30))
        #aug_color = iaa.ChangeColorTemperature((1100, 10000))
        aug_color = iaa.ChangeColorTemperature((10000, 40000))
        aug_gamma = iaa.GammaContrast((0.5, 2.0), per_channel=True)
        #img = aug_bright(images=img)
        img = aug_gamma(images=img)
        img = aug_color(images=img)
        return img.reshape(ih, iw, ic)
    elif domain_style == 2: ##blur and pooling
        #aug_gaus_blur = iaa.imgcorruptlike.GaussianBlur(severity=1)
        #aug_def_blur = iaa.imgcorruptlike.DefocusBlur(severity=2)
        #aug_mot_blur = iaa.imgcorruptlike.MotionBlur(severity=2)
        #aug = iaa.AllChannelsCLAHE()
        aug_color = iaa.ChangeColorTemperature((1000, 4000))
        img = aug_color(images=img)
        return img.reshape(ih, iw, ic)
    elif domain_style == 3: ##noise
        aug_gaus_noise = iaa.imgcorruptlike.GaussianNoise(severity=2)
        #aug_imp_noise = iaa.imgcorruptlike.ImpulseNoise(severity=2)
        img = aug_gaus_noise(images=img)
        return img.reshape(ih, iw, ic)
'''
def convert_style(img, domain_style=0):
    ih, iw, ic = img.shape
    img = img.reshape(1, ih, iw, ic)
    if domain_style == 0:   ##origin
        return img.reshape(ih, iw, ic)
    elif domain_style == 1: ##color and contrast style
        aug_bright = iaa.AddToBrightness((-30,30))
        #aug_color = iaa.ChangeColorTemperature((1100, 10000))
        aug_color = iaa.ChangeColorTemperature((8000, 10000))
        aug_gamma = iaa.GammaContrast((0.5, 2.0), per_channel=True)
        aug_gaus_noise = iaa.imgcorruptlike.GaussianNoise(severity=2)
        #img = aug_bright(images=img)
        img = aug_gamma(images=img)
        img = aug_color(images=img)
        img = aug_gaus_noise(images=img)
        return img.reshape(ih, iw, ic)
    elif domain_style == 2: ##blur and pooling
        aug_gaus_blur = iaa.imgcorruptlike.GaussianBlur(severity=2)
        #aug_def_blur = iaa.imgcorruptlike.DefocusBlur(severity=2)
        #aug_mot_blur = iaa.imgcorruptlike.MotionBlur(severity=2)
        img = aug_gaus_blur(images=img)
        return img.reshape(ih, iw, ic)
    elif domain_style == 3: ##noise
        aug_gaus_noise = iaa.imgcorruptlike.GaussianNoise(severity=2)
        #aug_imp_noise = iaa.imgcorruptlike.ImpulseNoise(severity=2)
        img = aug_gaus_noise(images=img)
        return img.reshape(ih, iw, ic)
'''
    
def collate_fn(batch):
    """
    Args:
        batch: list of tuple, 0 is images, 1 is img_meta, 2 is target
    Returns:
    """
    batch = list(zip(*batch))
    imgs = batch[0]
    img_metas = batch[1]
    targets = batch[2]
    sc_imgs = batch[3]
    sc_tran_dicts = batch[4]
    tr_imgs = batch[5]
    tr_tran_dicts = batch[6]
    
    
    if len(imgs) == 1:
        batched_imgs = imgs[0].unsqueeze_(0)
    else:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))

        batch_shape = (len(imgs),) + max_size
        batched_imgs = imgs[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(imgs, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    
    if len(sc_imgs) == 1:
        batched_sc_imgs = sc_imgs[0].unsqueeze_(0)
    else:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in sc_imgs]))

        batch_shape = (len(sc_imgs),) + max_size
        batched_sc_imgs = sc_imgs[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(sc_imgs, batched_sc_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    
    if len(tr_imgs) == 1:
        batched_tr_imgs = tr_imgs[0].unsqueeze_(0)
    else:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tr_imgs]))

        batch_shape = (len(tr_imgs),) + max_size
        batched_tr_imgs = tr_imgs[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(tr_imgs, batched_tr_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    return batched_imgs.contiguous(), img_metas, targets, batched_sc_imgs.contiguous(), sc_tran_dicts, batched_tr_imgs.contiguous(), tr_tran_dicts
    #return batched_imgs.contiguous(), img_metas, targets, img_metas, img_metas, img_metas, img_metas
