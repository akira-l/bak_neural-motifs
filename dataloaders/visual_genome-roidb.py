"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
import pickle
import copy

from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from collections import defaultdict
from pycocotools.coco import COCO
import pdb

class VG(Dataset):
    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=VG_SGG_DICT_FN,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=20,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 use_proposals=False):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode

        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'

        self.filenames = load_image_filenames(image_file)

        use_roidb = False
        if use_roidb==False:
            self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(self.filenames,
                            self.roidb_file,
                            self.mode,
                            num_im,
                            num_val_im=num_val_im,
                            filter_empty_rels=filter_empty_rels,
                            filter_non_overlap=(self.filter_non_overlap and self.is_train))
            #self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)
            self.ind_to_classes = [str(i) for i in range(401)]
            self.ind_to_predicates = [str(i) for i in range(501)]
        else:
            self.roidb_file = './roidb.pkl'
            self.gt_dict_file = './gt_data-cls400-rela90-hardsample.pt'
            self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_roidb_graphs(self.roidb_file, self.gt_dict_file, self.mode, num_im, num_val_im=num_val_im, filter_empty_rels=filter_empty_rels, filter_non_overlap=self.filter_non_overlap and self.is_train)
            self.ind_to_classes = [str(i) for i in range(401)]
            self.ind_to_predicates = [str(i) for i in range(501)]



        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
        #self.reserve_name, self.reserve_idx = self.check_filename(self.filenames)


        if use_proposals:
            #print("Loading proposals", flush=True)
            p_h5 = h5py.File(PROPOSAL_FN, 'r')
            rpn_rois = p_h5['rpn_rois']
            rpn_scores = p_h5['rpn_scores']
            rpn_im_to_roi_idx = np.array(p_h5['im_to_roi_idx'][self.split_mask])
            rpn_num_rois = np.array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np.column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        #self.check(rpn_im_to_roi_idx, rpn_num_rois)

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]])

        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test


    def check_filename(self, filename):
        namelist = torch.load('./namelist.pt')
        namelist_ = [tmp.split('/')[-1] for tmp in namelist]
        reserve_list = []
        reserve_idx = []
        for name_item in filename:
            if name_item.split('/')[-1] in namelist_:
                reserve_list.append(name_item)
                reserve_idx.append(filename.index(name_item))

        return reserve_list, reserve_idx

    def __getitem__(self, index):
        image_unpadded = Image.open(self.filenames[index]).convert('RGB')
        #index = self.reserve_idx[index]

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        gt_boxes = self.gt_boxes[index].copy()

        # Boxes are already at BOX_SCALE
        if self.is_train:
            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1])
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0])

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        w, h = image_unpadded.size
        box_scale_factor = BOX_SCALE / max(w, h)

        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max(w, h)
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        gt_rels = self.relationships[index].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)

        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': gt_rels,
            'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
            'index': index,
            'flipped': flipped,
            'fn': self.filenames[index],
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.gt_boxes)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
    print('\n\n-----notifiy--------\n', len(fns))
    assert len(fns) == 108073#108249
    return fns


def load_roidb_graphs(roidb_path, gt_dict_path, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True, filter_non_overlap=False):

    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    print('load roidb ....')
    roidb_file = open(roidb_path, 'rb')
    roidb = pickle.load(roidb_file, encoding='bytes')
    roidb_file.close()
    print('roidb loading done')

    gt_dict, cls_cluster = torch.load(gt_dict_path)
    cls_cluster = np.array(cls_cluster)#np.insert(np.array(cls_cluster)+1, 0, 0)
    cls_cluster_ = np.insert(np.array(cls_cluster)+1, 0, 0)
    rela_gather = []
    box_gather = []
    cls_gather = []
    name_gather = []

    for roidb_sub in roidb:
        rela = copy.deepcopy(roidb_sub[b'gt_relations'])
        rela_bak = copy.deepcopy(roidb_sub[b'gt_relations'])
        cls_ = copy.deepcopy(roidb_sub[b'gt_classes'])
        name_ = copy.deepcopy(roidb_sub[b'image'])
        name_ = str(name_)[2:-1].split('/')[-1]
        rela[:, 0] = cls_cluster[cls_[rela[:, 0]] -1]
        rela[:, 2] = cls_cluster[cls_[rela[:, 2]] -1]
        rela[:, 1] -= 1
        rela_list = rela.tolist()
        rela_bak_list = rela_bak.tolist()
        reserve_flag = False
        rela_data = []
        rela_src_data = []
        for rela_item, rela_src_item in zip(rela_list, rela_bak_list):
            rela_key = '-'.join([str(x) for x in rela_item])
            if rela_key in gt_dict:
                reserve_flag = True
                rela_data.append(rela_item)
                rela_src_data.append(rela_src_item)
        if reserve_flag:
            reserve_rela = np.array(rela_src_data)
            # swap for matching
            tmp = reserve_rela[:, 1]
            reserve_rela[:, 1] = reserve_rela[:, 2]
            reserve_rela[:, 2] = tmp
            reserve_cls = cls_cluster_[cls_]
            reserve_box = copy.deepcopy(roidb_sub[b'boxes'])
            rela_gather.append(reserve_rela)
            box_gather.append(reserve_box)
            cls_gather.append(reserve_cls)
            name_gather.append(name_)

    #mask_gather = [True for x in range(len(rela_gather)-num_val_im)] + \
    #             [False for x in range(num_val_im)]
    return box_gather, cls_gather, rela_gather, name_gather


def load_graphs(file_name, graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roidb_file = './roidb.pkl'
    gt_dict_file = './gt_data-cls400-rela90-hardsample.pt'
    box_gather, cls_gather, rela_gather, name_gather = load_roidb_graphs(roidb_file,
                  gt_dict_file,
                  mode,
                  num_im,
                  num_val_im=num_val_im,
                  filter_empty_rels=filter_empty_rels,
                  filter_non_overlap=filter_non_overlap)

    file_name_split = [tmp.split('/')[-1] for tmp in file_name]
    split_mask_bool = [True if tmp in file_name_split else False for tmp in name_gather]
    #for tmp in name_gather:


    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask_bool)[0]
    test_num = 100
    if mode == 'test':
        image_index = image_index[-test_num:]
    else:
        image_index = image_index[:-test_num]

    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]
    boxes = np.array(box_gather)[image_index].tolist()
    boxes = [np.int32(tmp) for tmp in boxes]
    gt_classes = np.array(cls_gather)[image_index].tolist()
    relationships = np.array(rela_gather)[image_index].tolist()

    # recheck
    for box_tmp in boxes:
        if box_tmp.any()<0:
            print('box has bug')
            pdb.set_trace()
    for cls_tmp in gt_classes:
        if cls_tmp.any()<0:
            print('cls has bug')
            pdb.set_trace()
    for rela_tmp in relationships:
        if rela_tmp.any()<0:
            print('rela has bug')
            pdb.set_trace()

    print('all clear ')


    return split_mask_bool, boxes, gt_classes, relationships

    #split_mask = np.zeros_like(data_split).astype(bool)
    #split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)


    return split_mask, boxes, gt_classes, relationships


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs
        )
        return train_load, val_load



#
