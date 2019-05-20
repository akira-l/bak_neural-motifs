import os,sys
import numpy as np

import cv2
import torch

import pdb


def bbox_intersections(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_intersections_np(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_overlaps(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_overlaps_np(box_a, box_b)

    inter = bbox_intersections(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) *
              (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) *
              (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def make_corner(rela, boxes, cls, input_size, output_size, categories=50):
    # generate corner groud truth 
    # (subject_id, rela_num, corner_map)
    inds = cls[:, 0].data
    obj_tl_mask = np.zeros((len(inds), categories, output_size[0], output_size[1]), dtype=np.float32)
    obj_br_mask = np.zeros((len(inds), categories, output_size[0], output_size[1]), dtype=np.float32)
    obj_tl_offset = np.zeros((len(inds), 2), dtype=np.float32)
    obj_br_offset = np.zeros((len(inds), 2), dtype=np.float32)
    gt_inds = np.zeros((len(inds), ), dtype=np.float32)
    width_ratio = output_size[0] / input_size[0]
    height_ratio = output_size[1] /input_size[1]

    box_offsets = boxes
    box_offsets[:, [0, 2]] *= box_offsets*width_ratio
    box_offsets[:, [1, 3]] *= box_offsets*height_ratio
    box_offsets[:, [0, 2]] -= int(box_offsets[:, [0, 2]])
    box_offsets[:, [1, 3]] -= int(box_offsets[:, [1, 3]])

    if gaussian_rad == -1:
        radius = gaussian_radius((height, width), gaussian_iou)
        radius = max(0, int(radius))
    else:
        radius = gaussian_rad

    for img_counter in range(max(inds)):
        skip_num = torch.nonzero(img_counter < inds).size(0)
        img_inds = torch.nonzero(img_counter == inds).view(-1)
        fetch_box = boxes[img_inds]
        rela_sub = rela[torch.nonzero(img_counter == rela[:, 0]).data.view(-1)]

        for fetch_rela in rela_sub:
            gt_inds[skip_num+fetch_rela[0]] = 1.0
            draw_gaussian(obj_tl_mask[skip_num+fetch_rela[0], fetch_rela[2], :, :], 
                          [fetch_box[fetch_rela[1]][0], fetch_box[fetch_rela[1]][1]],
                          radius)
            draw_gaussian(obj_br_mask[skip_num+fetch_rela[0], fetch_rela[2], :, :], 
                          [fetch_box[fetch_rela[1]][2], fetch_box[fetch_rela[1]][3]], 
                          radius)
            obj_tl_offset[skip_num+fetch_rela[0], :] = box_offsets[fetch_rela[1], [0, 1]]
            obj_br_offset[skip_num+fetch_rela[0], :] = box_offsets[fetch_rela[1], [2, 3]]
    return gt_inds, obj_tl_mask, obj_br_mask, obj_tl_offset, obj_br_offset



'''
(Pdb) dir(b)
['__class__', '__delattr__', '__dict__', '__dir__', 
'__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', 
'__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__',
 '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
  '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
   '__subclasshook__', '__weakref__', '_chunkize', '_scatter',
    'all_anchor_inds', 'all_anchors', 'anchor_chunks', 'append',
     'batch_size', 'batch_size_per_gpu', 'gt_box_chunks',
      'gt_boxes', 'gt_boxes_primary', 'gt_classes',
       'gt_classes_primary', 'gt_nodes', 'gt_rels', 'gt_sents',
        'im_sizes', 'imgs', 'is_flickr', 'is_rel', 'is_train',
         'mode', 'num_gpus', 'primary_gpu', 'proposal_chunks',
          'proposals', 'reduce', 'scatter', 'sent_lengths',
           'train_anchor_inds', 'train_anchor_labels',
             'train_anchors', 'train_chunks', 'volatile']
'''