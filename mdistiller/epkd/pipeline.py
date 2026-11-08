"""CAM/window/augmentation behavior adapted from Soobiwan/epkd-vision@1e75cd1.

Only stage coordinates are cached in memory; no teacher backward or persistent crop bank.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from ..distillers.SimKD import classifier, final_feature


@torch.no_grad()
def teacher_cam(teacher, image, target):
    teacher.eval()
    logits, features = teacher(image)
    feature = final_feature(features)
    head = classifier(teacher)
    if not isinstance(head, nn.Linear):
        raise ValueError("CAM requires an audited GAP + linear teacher head")
    rebuilt = head(feature.mean((2, 3)))
    torch.testing.assert_close(logits, rebuilt, rtol=1e-4, atol=1e-5,
                               msg="Selected CAM features do not reconstruct teacher logits")
    cam = torch.einsum("bchw,bc->bhw", feature.float(), head.weight[target].float()).clamp_min(0).unsqueeze(1)
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam /= cam.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1, 1)
    return logits, cam


def top_windows(cam, size, stride=8, count=3):
    mass = F.avg_pool2d(cam, size, stride=stride)
    k = min(count, mass[0].numel())
    top = mass.flatten(1).topk(k, 1).indices
    # Match EP-KD: repeat the last available rank if fewer than three windows exist.
    top = top[:, torch.arange(count, device=cam.device).clamp_max(k-1)]
    gw = mass.size(-1)
    return torch.stack(((top // gw)*stride, (top % gw)*stride), dim=-1)


class ImageViews:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def normalize(self, raw):
        return (raw.float()/255. - self.mean) / self.std

    @staticmethod
    def crop(image, row, col, size):
        ar = torch.arange(image.size(0), device=image.device)[:, None, None]
        rr = row[:, None, None] + torch.arange(size, device=image.device)[None, :, None]
        cc = col[:, None, None] + torch.arange(size, device=image.device)[None, None, :]
        return image.permute(0, 2, 3, 1)[ar, rr, cc].permute(0, 3, 1, 2).contiguous()

    def augment(self, image, padding):
        size, batch = image.shape[-1], len(image)
        if padding:
            padded = F.pad(image, (padding,)*4, mode="reflect")
            offset = torch.randint(0, 2*padding+1, (batch, 2), device=image.device)
            image = self.crop(padded, offset[:, 0], offset[:, 1], size)
        flip = torch.rand(batch, device=image.device) < .5
        return torch.where(flip[:, None, None, None], image.flip(-1), image)

    def full(self, raw):
        return self.augment(self.normalize(raw), 4)

    def view(self, raw, coordinates, size, member):
        pos = coordinates[:, member]
        image = self.crop(self.normalize(raw), pos[:, 0], pos[:, 1], size)
        return self.augment(image, 2)


@torch.no_grad()
def stage_coordinates(teacher, raw, target, views, size, stride=8, batch_size=256):
    coordinates = []
    for start in range(0, len(raw), batch_size):
        end = start + batch_size
        _, cam = teacher_cam(teacher, views.normalize(raw[start:end]), target[start:end])
        coordinates.append(top_windows(cam, size, stride))
    return torch.cat(coordinates)


@contextmanager
def crop_batchnorm(distiller):
    """Keep running mean/variance from full images, including trainable alignment modules."""
    frozen = {id(m) for m in distiller.teacher.modules()}
    modules = [m for m in distiller.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and id(m) not in frozen]
    saved = [m.momentum for m in modules]
    try:
        for m in modules:
            m.momentum = 0.
        yield
    finally:
        for m, momentum in zip(modules, saved):
            m.momentum = momentum
