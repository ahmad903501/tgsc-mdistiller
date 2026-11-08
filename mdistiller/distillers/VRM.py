"""VRM losses ported unchanged from VISION-SJTU/VRM@56d02ce.

This is the published code variant: it does not implement the paper's entropy pruning.
"""
import torch
from torch.nn import functional as F
from contextlib import nullcontext
from ._base import Distiller

def isv_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50):  # f_s: (B,100),  f_t: (B,100)
    stu_w = f_sw.view(f_sw.shape[0], -1); stu_s = f_ss.view(f_ss.shape[0], -1) # (B,100)
    tea_w = f_tw.view(f_tw.shape[0], -1); tea_s = f_ts.view(f_ts.shape[0], -1) # (B,100)

    ########### ZSNorm of Logits ###########
    stu_w_mean = stu_w.mean(dim=-1, keepdim=True); stu_s_mean = stu_s.mean(dim=-1, keepdim=True) # (B,1) # mean is sample-wise
    tea_w_mean = tea_w.mean(dim=-1, keepdim=True); tea_s_mean = tea_s.mean(dim=-1, keepdim=True) # (B,1)
    stu_w_std = stu_w.std(dim=-1, keepdim=True); stu_s_std = stu_s.std(dim=-1, keepdim=True) # (B,1) # std is sample-wise
    tea_w_std = tea_w.std(dim=-1, keepdim=True); tea_s_std = tea_s.std(dim=-1, keepdim=True) # (B,1)
    stu_w = (stu_w - stu_w_mean) / (1e-7 + stu_w_std); stu_s = (stu_s - stu_s_mean) / (1e-7 + stu_s_std)
    tea_w = (tea_w - tea_w_mean) / (1e-7 + tea_w_std); tea_s = (tea_s - tea_s_mean) / (1e-7 + tea_s_std)

    ########### Convert Logits to Probs ###########
    T = 4.0
    stu_w = F.softmax(stu_w / T, dim=1); stu_s = F.softmax(stu_s / T, dim=1)
    tea_w = F.softmax(tea_w / T, dim=1); tea_s = F.softmax(tea_s / T, dim=1)

    ########## RKD Angle loss ##########
    with torch.no_grad():
        # compute pairwise L1 distance within batch
        td = tea_w.unsqueeze(0) - tea_s.unsqueeze(1) # (B,B,256)
        norm_td = F.normalize(td, p=2, dim=2) # (B,B,256) - normalise dim 2 using L2 distance
    # compute pairwise L1 distance within batch
    sd = stu_w.unsqueeze(0) - stu_s.unsqueeze(1) # (B,B,256)
    norm_sd = F.normalize(sd, p=2, dim=2) # (B,B,256)
    loss_isv = F.smooth_l1_loss(norm_td, norm_sd) * edge_weight # (B,B,256)
    return loss_isv

def icv_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50):  # f_s: (B,100),  f_t: (B,100)
    stu_w = f_sw.view(f_sw.shape[0], -1); stu_s = f_ss.view(f_ss.shape[0], -1) # (B,100)
    tea_w = f_tw.view(f_tw.shape[0], -1); tea_s = f_ts.view(f_ts.shape[0], -1) # (B,100)

    ########### ZSNorm of Logits ###########
    stu_w_mean = stu_w.mean(dim=-1, keepdim=True); stu_s_mean = stu_s.mean(dim=-1, keepdim=True) # (B,1) # mean is sample-wise
    tea_w_mean = tea_w.mean(dim=-1, keepdim=True); tea_s_mean = tea_s.mean(dim=-1, keepdim=True) # (B,1)
    stu_w_std = stu_w.std(dim=-1, keepdim=True); stu_s_std = stu_s.std(dim=-1, keepdim=True) # (B,1) # std is sample-wise
    tea_w_std = tea_w.std(dim=-1, keepdim=True); tea_s_std = tea_s.std(dim=-1, keepdim=True) # (B,1)
    stu_w = (stu_w - stu_w_mean) / (1e-7 + stu_w_std); stu_s = (stu_s - stu_s_mean) / (1e-7 + stu_s_std)
    tea_w = (tea_w - tea_w_mean) / (1e-7 + tea_w_std); tea_s = (tea_s - tea_s_mean) / (1e-7 + tea_s_std)

    ########### Convert Logits to Probs ###########
    T = 4.0
    stu_w = F.softmax(stu_w / T, dim=1); stu_s = F.softmax(stu_s / T, dim=1)
    tea_w = F.softmax(tea_w / T, dim=1); tea_s = F.softmax(tea_s / T, dim=1)

    ########## RKD Angle loss ##########
    with torch.no_grad():
        # compute pairwise L1 distance within batch
        td = (tea_w.unsqueeze(2) - tea_s.unsqueeze(1)).permute(1, 2, 0) # (B,B,256)
        norm_td = F.normalize(td, p=2, dim=2) # (B,B,256) - normalise dim 2 using L2 distance
    # compute pairwise L1 distance within batch
    sd = (stu_w.unsqueeze(2) - stu_s.unsqueeze(1)).permute(1, 2, 0) # (B,B,256)
    norm_sd = F.normalize(sd, p=2, dim=2) # (B,B,256)
    loss_a = F.smooth_l1_loss(norm_td, norm_sd)
    loss_icv = edge_weight * loss_a
    return loss_icv

class VRM(Distiller):
    paired_views = True

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.teacher.eval().requires_grad_(False)

    def forward_train(self, image_weak, image_strong, target, crop_second=False, **kwargs):
        from ..epkd.pipeline import crop_batchnorm
        sw, _ = self.student(image_weak)
        # Native CAM crops use EP-KD running-stat handling;32px uses official VRM behavior.
        with crop_batchnorm(self) if crop_second else nullcontext():
            ss, _ = self.student(image_strong)
        with torch.no_grad():
            tw, _ = self.teacher(image_weak)
            ts, _ = self.teacher(image_strong)
        losses = {
            "loss_ce": F.cross_entropy(sw, target) + F.cross_entropy(ss, target),
            "loss_isv": 64.0 * isv_loss(sw, ss, tw, ts),
            "loss_icv": 32.0 * icv_loss(sw, ss, tw, ts),
        }
        return sw, losses
