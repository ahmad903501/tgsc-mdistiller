"""Exact CIFAR weak/strong augmentation from VISION-SJTU/VRM@56d02ce.

Only the required operations are vendored; no substitution by torchvision RandAugment.
"""
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from torchvision import transforms


class MultipleApply:
    """Apply a list of transformations to an image and get multiple transformed images.

    Args:
        transforms (list or tuple): list of transformations

    Example:

        >>> transform1 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224)
        ... ])
        >>> transform2 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224),
        ... ])
        >>> multiply_transform = MultipleApply([transform1, transform2])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]

def AutoContrast(img, _):
    return ImageOps.autocontrast(img)

def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)

def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, _):
    return ImageOps.equalize(img)

def Identity(img, v):
    return img

def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)

def Rotate(img, v):
    return img.rotate(v)

def Sharpness(img, v):
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v):
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def Solarize(img, v):
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)

def Cutout(img, v):
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)
        return img

def get_cifar100_train_transform_strong():
    train_transform_weak = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train_transform_strong = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_transform = MultipleApply([train_transform_weak, train_transform_strong])

    return train_transform
