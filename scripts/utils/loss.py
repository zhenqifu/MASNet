import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pytorch_ssim
import utils.pytorch_iou
# ssim_loss = utils.pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = utils.pytorch_iou.IOU(size_average=True)

bce_loss = nn.BCELoss()


def bce_ssim_loss(pred, target):

    bce_out = bce_loss(pred, target)
    # ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out

    return loss


def loss_task(output, labels):
    loss = bce_ssim_loss(output, labels)
    return loss


