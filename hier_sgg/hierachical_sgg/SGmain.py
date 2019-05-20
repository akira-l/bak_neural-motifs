# main class for scene graph generation 
import os, sys
import numpy as np

import torch

import pdb

class SGMain():
    def __init__():
        pass
    


if __name__ == '__main__':
    getgmap_module = CornerMap()

    proposal_num = 32
    ins_feat = torch.rand(proposal_num, 2048) # get from main detector 
    gt_map = getgmap_module(ins_feat, gt_package) # ground truth corner map should also get with faster rcnn 
    glo_feat = torch.rand(1, 2048)
    # model init 
    att_module = RelaAtt()
    corner_module = SgCorner()
    # merge attention : gcn / transformer / bilinear
    att_out = att_module(ins_feat, glo_feat, method='bilinear')
    corner_map, corner_loss = corner_module(att_out, gt_map)