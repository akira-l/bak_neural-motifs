
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch
import torchvision

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
import pdb

import cv2

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
if conf.test:
    val = test

ckpt = torch.load(conf.ckpt)

all_pred_entries = []

evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)

relation_dict = {"0": "none", "1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
object_dict = {"0": "unknown", "1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}

print("Found {}! Loading from it".format(conf.cache))


with open(conf.cache,'rb') as f:
    all_pred_entries = pkl.load(f)


pred_relation_count = np.zeros(51)
pred_relation_count_ep0 = np.zeros(51)
gt_relation_count = np.zeros(51)
for i, pred_entry in enumerate(tqdm(all_pred_entries)):
    gt_entry = {
        'gt_classes': val.gt_classes[i].copy(),
        'gt_relations': val.relationships[i].copy(),
        'gt_boxes': val.gt_boxes[i].copy(),
    }
    
    gt_boxes = val.gt_boxes[i]*(1/BOX_SCALE)
    gt_class = val.gt_classes[i]
    pred_boxes = pred_entry['pred_boxes']*(1/BOX_SCALE)
    gt_relation = val.relationships[i].tolist()
    pred_relation = pred_entry['rel_scores'].tolist()
    pred_relation_index = pred_entry['pred_rel_inds'].tolist()
    pred_cls = pred_entry['pred_classes']

    for obj1, obj2, gt_rel in gt_relation:
        # gt relation count 

        gt_relation_count[gt_rel] += 1


    for pred_obj, pred_rel in zip(pred_relation_index, pred_relation):

        relation_index = pred_rel.index(max(pred_rel))
        pred_relation_count[relation_index] += 1
        if relation_index == 0:
            pred_rel[0] = 0
            relation_index = pred_rel.index(max(pred_rel))
            pred_relation_count_ep0[relation_index] += 1
            #pdb.set_trace()

np.save('gt_relation_count.npy', gt_relation_count)
np.save('pred_relation_count.npy', pred_relation_count)
np.save('pred_relation_count_ep0.npy', pred_relation_count_ep0)



'''
grid = torchvision.utils.make_grid(sampled)
torchvision.utils.save_image(grid, './train/'+img_path[img_counter].split('/')[-1])
#boxes: List where each element is a [num_gt, 4] array of ground truth boxes (x1, y1, x2, y2)
'''