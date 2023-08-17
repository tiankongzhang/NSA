# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
from PIL import ImageFilter
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps
import imgaug.augmenters as iaa
from .augmentation import COLOR, CUTOUT, RANDOM_COLOR_POLICY_OPS
from .auto_augment import apply_policy

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, imgs):
        '''
        params = []
        for i in range(len(imgs)):
            params.append({})
        
        for a in self.augmentations:
            for i, (img, param) in enumerate(zip(imgs, params)):
                out = a(img)
                if out[1] is not None:
                   params[i][out[1].keys()[0]] = out[1][out[1].keys()[0]]
                imgs[i] = out[0]
        '''
        params = {}
        for a in self.augmentations:
            out = a(imgs)
            if out[1] is not None:
               for key, value in out[1].items():
                   #params[out[1].keys()[0]] = out[1][out[1].keys()[0]]
                   params[key] = value
            imgs = out[0]

        if self.PIL2Numpy:
            imgs = np.array(imgs)
            
        return imgs, params
'''
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        params = None
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, params
            
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                params
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        
        params = {}
        params['RandomCrop'] = (x1, y1, x1 + tw, y1 + th, tw, th)
    
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            params
        )
'''
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        params = None
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, params
        
            
        if w < tw or h < th:
           td = 0
           bd = 0
           
           ld = 0
           rd = 0
           if w < tw:
               ld = 0
               rd = tw - ld - w
               
           if h < th:
               td = 0
               bd = th - td - h
               
           
           img = ImageOps.expand(img, border=(ld, td, rd, bd), fill=0)
           
           return img,params

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        
        params = {}
        params['RandomCrop'] = (x1, y1, x1 + tw, y1 + th, tw, th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            params
        )

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        params = {}
        
        w, h = img.size
        th, tw = self.size
        #print(w,h, th, tw)
        td = 0
        bd = 0
        
        ld = 0
        rd = 0
        if w < tw:
            ld = int((tw - w) / 2)
            rd = tw - ld - w
            
        if h < th:
            td = int((th - h) / 2)
            bd = th - td - h
            
        
        img = ImageOps.expand(img, border=(ld, td, rd, bd), fill=0)
        
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        
        params['CenterCrop'] = (x1-ld, y1-td, x1 - ld + tw, y1 - td + th, tw, th)
        
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            params
        )


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img):
        params = None
        return (tf.adjust_gamma(img, random.uniform(1 - self.gamma, 1 + self.gamma)),
                params
                )


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img):
        return (tf.adjust_saturation(img,
                                    random.uniform(1 - self.saturation,
                                                   1 + self.saturation)), None)


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img):
        return (tf.adjust_hue(img, random.uniform(-self.hue,
                                                  self.hue)), None)


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img):
        return (tf.adjust_brightness(img,
                                    random.uniform(1 - self.bf,
                                                   1 + self.bf)), None)

class AdjustSharpness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img):
        return (tf.adjust_sharpness(img,
                                    random.uniform(1 - self.bf,
                                                   1 + self.bf)), None)

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img):
        return (tf.adjust_contrast(img,
                                  random.uniform(1 - self.cf,
                                                 1 + self.cf)), None)
class AdjustEqualize(object):
    def __init__(self, cf=None):
        self.cf = cf

    def __call__(self, img):
        return (tf.equalize(img), None)

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        params=None
        if random.random() < self.p:
            params = {}
            params['RHF'] = (img.size)
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                #img.flip(),
                params
            )
                
        return img, params


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        params=None
        if random.random() < self.p:
            params = {}
            params['RVF'] = (img.size)
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                params
            )
        return img, params

'''
class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )
'''
class RandomTranslate(object):
    def __init__(self, offset, p=0.8):
        self.offset = offset # tuple (delta_x, delta_y)
        self.p = p

    def __call__(self, img):
        params = {}
        random_prob = random.random() - 0.5
        if random_prob > 0.0:
            fh = 1.0
        else:
            fh = -1.0
        x_offset = int(1 * random_prob * self.offset[1]) + int(fh * self.offset[0])
        y_offset = int(1 * random_prob * self.offset[1]) + int(fh * self.offset[0])
        
        params['Translate'] = [abs(x_offset), abs(y_offset), x_offset, y_offset]
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img,
                              y_crop_offset,
                              x_crop_offset,
                              img.size[1]-abs(y_offset),
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        return tf.pad(cropped_img,
                     padding_tuple,
                     padding_mode='reflect'), params


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        params = None
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0), params)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size, scale_sz=[0.5, 1.1]):
        self.size = size
        self.crop = RandomCrop(self.size)
        self.scale_sz = scale_sz

    def __call__(self, img):
        params={}

        prop = 1.0 * img.size[0] / img.size[1]
        w = int(random.uniform(self.scale_sz[0], self.scale_sz[1]) * self.size)
        #w = self.size
        h = int(w/prop)
        
        sh = h * 1.0 / img.size[1]
        sw = w * 1.0 / img.size[0]
        params['RandomSized'] = (sw, sh)

        img = img.resize((w, h), Image.BILINEAR)

        return img, params
        
class RandomLSized(object):
    def __init__(self, size, scale_sz=[0.5, 1.1]):
        self.size = size
        self.crop = RandomCrop(self.size)
        self.scale_sz = scale_sz

    def __call__(self, img):
        params={}

        prop = 1.0 * img.size[0] / img.size[1]
        
        if len(self.scale_sz) == 2:
             scsz = random.uniform(self.scale_sz[0], self.scale_sz[1])
        else:
             if random.random() > 0.3:
                 scsz = random.uniform(self.scale_sz[0], self.scale_sz[1])
             else:
                 scsz = random.uniform(self.scale_sz[1], self.scale_sz[2])
                 
        if random.random() > 0.5:
            scsz = scsz
        else:
            scsz = 1.0 / (scsz+1e-5)
            
        w = int( scsz * self.size)
        #w = self.size
        h = int(w/prop)
        
        sh = h * 1.0 / img.size[1]
        sw = w * 1.0 / img.size[0]
        params['RandomSized'] = (sw, sh)

        img = img.resize((w, h), Image.BILINEAR)

        return img, params

class RandomRSized(object):
    def __init__(self, size, scale_sz=[0.5, 1.1]):
        self.size = size
        self.crop = RandomCrop(self.size)
        self.scale_sz = scale_sz

    def __call__(self, img):
        params={}

        prop = 1.0 * img.size[0] / img.size[1]
        
        if len(self.scale_sz) == 2:
             scsz = random.uniform(self.scale_sz[0], self.scale_sz[1])
        else:
             if random.random() > 0.3:
                 scsz = random.uniform(self.scale_sz[0], self.scale_sz[1])
             else:
                 scsz = random.uniform(self.scale_sz[1], self.scale_sz[2])
            
        w = int( scsz * self.size)
        #w = self.size
        h = int(w/prop)
        
        sh = h * 1.0 / img.size[1]
        sw = w * 1.0 / img.size[0]
        params['RandomSized'] = (sw, sh, scsz)

        img = img.resize((w, h), Image.BILINEAR)

        return img, params

'''
class RandGreyscale:
    def __init__(self, p = 0.1):
        self.p = p

    def __call__(self, image):
        params = None
    
        if self.p > random.random():
            image = tf.to_grayscale(image, num_output_channels=3)

        return image, params

'''
class IaRandGaussianNoise_v0:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, severity=2, p=0.3):
        self.severity = severity
        self.p = p

    def __call__(self, image):
        params = None
        if random.random() < self.p:
            severity = int(random.uniform(1, self.severity))
            aug_gaus_noise = iaa.imgcorruptlike.GaussianNoise(severity=severity)
            image = np.array(image, dtype=np.uint8)
            ih, iw, ic = image.shape
            image = image.reshape(1, ih, iw, ic)
            image = aug_gaus_noise(images=image)
            image  = Image.fromarray(image.reshape(ih, iw, ic))

        return image, params

class IaRandGaussianNoise_v1:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, severity=2, p=0.3):
        self.severity = severity
        self.p = p

    def __call__(self, image):
        params = None
        if random.random() < self.p:
            aav = random.random() * 0.15
            #aug_gaus_noise = iaa.AdditiveGaussianNoise(scale=aav*255, per_channel=True)
            aug_gaus_noise = iaa.AdditiveGaussianNoise(scale=aav*255)
            
            image = np.array(image, dtype=np.uint8)
            ih, iw, ic = image.shape
            image = image.reshape(1, ih, iw, ic)
            image = aug_gaus_noise(images=image)
            image  = Image.fromarray(image.reshape(ih, iw, ic))

        return image, params

class IaRandImpulseNoise:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, severity=2, p=0.3):
        self.severity = severity
        self.p = p

    def __call__(self, image):
        params = None
        #print(np.array(image, dtype=np.uint8))
        if random.random() < self.p:
            aug_imp_noise = iaa.imgcorruptlike.ImpulseNoise(severity=1)
            
            image = np.array(image, dtype=np.uint8)
            ih, iw, ic = image.shape
            image = image.reshape(1, ih, iw, ic)
            image = aug_imp_noise(images=image)
            image  = Image.fromarray(image.reshape(ih, iw, ic))

        return image, params

class IaRandImpulseNoise:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, severity=2, p=0.3):
        self.severity = severity
        self.p = p

    def __call__(self, image):
        params = None
        if random.random() < self.p:
            aug_imp_noise = iaa.imgcorruptlike.ImpulseNoise(severity=1)
            
            image = np.array(image, dtype=np.uint8)
            ih, iw, ic = image.shape
            image = image.reshape(1, ih, iw, ic)
            image = aug_imp_noise(images=image)
            image  = Image.fromarray(image.reshape(ih, iw, ic))

        return image, params

class IaRandColor:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image):
        params = None
        if random.random() < self.p:
            stx = random.randint(1100, 34000)
            aug_bright = iaa.AddToBrightness((-30,30))
            aug_color = iaa.ChangeColorTemperature((stx, stx+4000))
            aug_gamma = iaa.GammaContrast((0.5, 2.0), per_channel=True)
            
            image = np.array(image, dtype=np.uint8)
            ih, iw, ic = image.shape
            image = image.reshape(1, ih, iw, ic)
            
            #img = aug_bright(images=img)
            img = aug_gamma(images=image)
            img = aug_color(images=image)
            image  = Image.fromarray(image.reshape(ih, iw, ic))

        return image, params

class RandGaussianBlur:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, radius=[.1, 2.], p=0.4):
        self.radius = radius
        self.p = p

    def __call__(self, image):
        params = None
        if random.random() < self.p:
            radius = random.uniform(self.radius[0], self.radius[1])
            image = image.filter(ImageFilter.GaussianBlur(radius))

        return image, params


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
        
class RandColorJilter:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, pms=[], p=0.4):
        self.pms = pms
        self.p = p
        
        augmentation = []
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        '''
        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)
        '''
        self.utransforms = transforms.Compose(augmentation)

    def __call__(self, image):
        params = None
        image = self.utransforms(image)

        return image, params


class Iaaa_Jitter:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self,):
        self.jitter_aug_op = COLOR

    def __call__(self, image):
        params = None
        
        image = np.array(image, dtype=np.uint8)
        ih, iw, ic = image.shape
        image = image.reshape(1, ih, iw, ic)
        
        image = self.jitter_aug_op(images=image)
        image  = Image.fromarray(image.reshape(ih, iw, ic))
        return image, params


class Iaaa_Cutout:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self,):
        self.cutout_op = CUTOUT

    def __call__(self, image):
        params = None
        
        image = np.array(image, dtype=np.uint8)
        ih, iw, ic = image.shape
        image = image.reshape(1, ih, iw, ic)
        image = self.cutout_op(images=image)
        image  = Image.fromarray(image.reshape(ih, iw, ic))

        return image, params

class Iaaa_Color:
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, p=1.0, magnitude=10):
        self.p = p
        self.magnitude = magnitude

    def __call__(self, image):
        params = None
        
        policy = lambda: [(op, self.p, np.random.randint(1, self.magnitude))  # pylint: disable=g-long-lambda
                          for op in np.random.choice(RANDOM_COLOR_POLICY_OPS, 1)]
        
        image = np.array(image, dtype=np.float32)
        ih, iw, ic = image.shape
        image = image.reshape(1, ih, iw, ic)
        
        
        image = self.numpy_apply_policies((image, policy()))
        #print(image.shape, ih, iw, ic)
        image = np.array(image, dtype=np.uint8)
        image  = Image.fromarray(image)
        return image, params
    
    def numpy_apply_policies(self, arglist):
        x, policies = arglist
        
        y_a = apply_policy(policies, self.normaize(x[0]))
        y_a = np.reshape(y_a, x[0].shape)
        y_a = self.unnormaize(y_a)
        
        #print(len(y_a), y_a[0].shape)
        return y_a
    
    def normaize(self, x):
      x /= 255.0
      x = x / 0.5 - 1.0
      return x

    def unnormaize(self, x):
      x = (x + 1.0) * 0.5 * 255.0
      return x



