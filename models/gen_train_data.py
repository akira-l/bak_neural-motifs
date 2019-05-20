from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
import pdb
import math

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau

conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

bbox = train.gt_boxes
obj_cls = train.gt_classes
rels = train.relationships

train_input = []
train_gt = []
counter = 0
for relation, boxes, clses in zip(rels, bbox, obj_cls):
    for sub_rels in relation:
        obj1_box = boxes[sub_rels[0]]
        obj1 = clses[sub_rels[0]]
        obj2_box = boxes[sub_rels[1]]
        obj2 = clses[sub_rels[1]]
        cur_rels = sub_rels[2]

        #pdb.set_trace()
        obj = [obj1, obj2]
        pos_emb = [(obj1_box[2]-obj1_box[0])/1024, 
                (obj1_box[3]-obj1_box[1])/1024, 
                (obj2_box[2]-obj2_box[0])/1024, 
                (obj2_box[3]-obj2_box[1])/1024, 
                ((obj1_box[0]+obj1_box[2])/2 - (obj2_box[0]+obj2_box[2])/2)/(obj2_box[2]-obj2_box[0]),
                ((obj1_box[2]+obj1_box[3])/2 - (obj2_box[2]+obj2_box[3])/2)/(obj2_box[3]-obj2_box[1]),
                (((obj1_box[0]+obj1_box[2])/2 - (obj2_box[0]+obj2_box[2])/2)/(obj2_box[2]-obj2_box[0]))**2,
                (((obj1_box[1]+obj1_box[3])/2 - (obj2_box[1]+obj2_box[3])/2)/(obj2_box[3]-obj2_box[1]))**2,
                math.log((obj1_box[2]-obj1_box[0])/(obj2_box[2]-obj2_box[0])+10**-5),
                math.log((obj1_box[3]-obj1_box[1])/(obj2_box[3]-obj2_box[1])+10**-5)]
        train_input.append([obj, pos_emb, cur_rels])
np.save('train.npy', train_input)


pdb.set_trace()



