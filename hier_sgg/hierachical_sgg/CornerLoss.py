import os, sys
import numpy as np

import torch
import torch.nn as nn

import pdb


def nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def focal_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    if tag_mean.dim() == 1:
        tag_mean = tag_mean.unsqueeze(0)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss



class CornerLoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1):
        super(CornerLoss, self).__init__()
        
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight

    def corner_focal_loss(self, pred, gt, pred_inds):
        return focal_loss(pred[pred_inds[0, :]], gt[pred_inds[1, :]])

    def tag_loss(self, pred_tl_tag, pred_br_tag, pred_inds, tl_mask, br_mask):
        get_tag_inds = pred_inds[0, np.where(pred_inds[2, :]==1, 1)]
        tl_tag = pred_tl_tag.view(-1)[tl_mask[get_tag_inds].eq(1).view(-1)]
        br_tag = pred_br_tag.view(-1)[br_mask[get_tag_inds].eq(1).view(-1)]

        num = tl_mask[get_tag_inds].eq(1).view(-1).sum()
        tag_mean = (tl_tag + br_tag) / 2
        tl_tag = torch.pow(tl_tag - tag_mean, 2) / (num + 1e-4)
        br_tag = torch.pow(br_tag - tag_mean, 2) / (num + 1e-4)
        pull = tl_tag.sum() + br_tag.sum()
        
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
        dist = 1 - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - 1 / (num + 1e-4)
        dist = dist / (num2 + 1e-4)
        dist = dist[mask]
        push = dist.sum()

        return pull, push 


    def offset_loss(self, pred_offset, gt_offset, gt_inds, pred_inds):
        get_tag_inds = pred_inds[0, np.where(pred_inds[2, :]==1, 1)]
        pred_offset_ = pred_offset.view(-1)[pred_inds[0, :]]
        gt_offset_ = gt_offset.view(-1)[pred_inds[1, :]]
        regr_loss = nn.functional.smooth_l1_loss(pred_offset, gt_offset, size_average=False)
        num = pred_inds[2, :].sum()
        return regr_loss / (num + 1e-4)



    def forward(self, outs, targets):
        pred_inds, pred_tl_heatmap, pred_br_heatmap, pred_tl_tag,\
             pred_br_tag, pred_tl_offset, pred_br_offset = outs
        gt_inds, tl_mask, br_mask, tl_offset, br_offset = targets

        # pred_inds should have 3* raws 
        # pred_inds[0, :] = rpn_inds use which rpn features
        # pred_inds[1, :] = corresponing ground truth index
        # pred_inds[2, :] = negtive or positive inds

        # pred heatmap should be rpn_number, rela_class, mask_x, mask_y
        pred_tl_heatmap = torch.sigmoid(pred_tl_heatmap)
        pred_br_heatmap = torch.sigmoid(pred_br_heatmap)

        loss = 0
        loss += self.corner_focal_loss(pred_tl_heatmap, tl_mask)
        loss += self.corner_focal_loss(pred_br_heatmap, br_mask)

        loss += self.tag_loss(pred_tl_tag, pred_br_tag, gt_inds, tl_mask, br_mask)

        loss += self.offset_loss(pred_tl_offset, tl_offset, gt_inds, pred_inds)
        loss += self.offset_loss(pred_br_offset, br_offset, gt_inds, pred_inds)

        return loss 





