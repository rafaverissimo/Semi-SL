import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image

def AutoContrast(img):
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    tensor = tensor.to(torch.uint8)
    t_tensor = transforms.functional.autocontrast(tensor)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img


def Equalize(img):
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    tensor = tensor.to(torch.uint8)
    t_tensor = transforms.functional.equalize(tensor)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img

def Solarize(img):  # [0, 256]
    v = random.randint(0,255)
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    tensor = tensor.to(torch.uint8)
    t_tensor = transforms.functional.solarize(tensor, v)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img


def Posterize(img):  # [4, 8]
    v = random.randint(4,8)
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    tensor = tensor.to(torch.uint8)
    t_tensor = transforms.functional.posterize(tensor, v)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img


def Contrast(img):  # [0.1,1.9]
    v = random.randint(1,19)
    v = v/10
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    tensor = tensor.to(torch.uint8)
    t_tensor = transforms.functional.adjust_contrast(tensor, v)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img


def Brightness(img):  # [0.1,1.9]
    v = random.randint(1,19)
    v = v/10
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    t_tensor = transforms.functional.adjust_brightness(tensor, v)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img

def Sharpness(img):  # [0.1,1.9]
    v = random.randint(1,19)
    v = v/10
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    t_tensor = transforms.functional.adjust_sharpness(tensor, v)
    toImg = transforms.ToPILImage(mode="RGB")
    t_img = toImg(t_tensor)
    return t_img


def augment_list():
    l = [
        (AutoContrast),
        (Equalize),
        (Posterize),
        (Solarize),
        (Contrast),
        (Brightness),
        (Sharpness)
    ]

    return l

#
# class Lighting(object):
#     """Lighting noise(AlexNet - style PCA - based noise)"""
#
#     def __init__(self, alphastd, eigval, eigvec):
#         self.alphastd = alphastd
#         self.eigval = torch.Tensor(eigval)
#         self.eigvec = torch.Tensor(eigvec)
#
#     def __call__(self, img):
#         if self.alphastd == 0:
#             return img
#
#         alpha = img.new().resize_(3).normal_(0, self.alphastd)
#         rgb = self.eigvec.type_as(img).clone() \
#             .mul(alpha.view(1, 3).expand(3, 3)) \
#             .mul(self.eigval.view(1, 3).expand(3, 3)) \
#             .sum(1).squeeze()
#
#         return img.add(rgb.view(3, 1, 1).expand_as(img))


# class CutoutDefault(object):
#     """
#     Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
#     """
#     def __init__(self):
#         self.length = 16
#
#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)
#
#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)
#
#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img


class RandAugment:
    def __init__(self, n):
        self.n = n
        self.m = 7      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)

        return img
