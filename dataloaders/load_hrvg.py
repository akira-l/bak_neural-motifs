import json
import os, sys

import h5py
import numpy as np
import xml.etree.ElementTree as ET

import pdb



def gen_filenames_split():
    # generate filenames and split files
    # output .npy files
    # check all xml file, select files has relationships
    # make a relation-image dict
    # sample in dict to generate a split file

    obj_vocab, rela_vocab = load_hrvg_info()
    xml_folder = './converted_xml'
    xml_list = os.listdir(xml_folder)
    img_filename_list = [x.replace('xml', 'jpg') for x in xml_list]

    rela_img_dict = []
    for rela_name in rela_vocab:
        rela_img_dict[rela_name] = []

    # check xml make file list and relation-img dict
    for xml_name in xml_list:
        xml_path = os.path.join(xml_folder, xml_name)

        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        if xml_root[-1].tag != 'relation':
            continue

        for item_num in range(1, item_all_count):
            item = xml_root[-item_num]
            if not item.tag == 'relation':
                break
            assert item[2].tag == 'predicate'
            rela_img_dict[item[2].text].append(xml_name.replace('xml', 'jpg'))

    test_split = []
    for rela_name in rela_vocab:
        rela_img = rela_img_dict[rela_name]
        img_has_rela_num = len(rela_img)
        if img_has_rela_num < 3:
            continue
        else:
            test_split.extend(random.sample(rela_img,
                            int(img_has_rela_num*0.3)))

    test_split = list(set(test_split))
    if len(test_split) > int(0.3*len(img_filename_list)):
        test_split = random.sample(test_split, int(0.3*len(img_filename_list)))

    print('loading %d test images' % len(test_split))

    train_split = list(filter(lambda x: x not in test_split, img_filename_list))

    print('loading %d train images' % len(train_split))

    img_split = [0 if x in train_split else 2 for x in img_filename_list]

    print(' %d images in overall' % len(img_split))

    return np.array(img_split), img_filename_list


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
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

    data_split, filenames = gen_filenames_split()
    ind_to_classes, ind_to_predicates = load_hrvg_info()

    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    img_candidate = np.array(filenames)
    img_candidate = img_candidate[split_mask].tolist()

    cls_gather  = []
    rela_gather = []
    box_gather  = []
    for img_name in img_candidate:
        xml_name = img_name.replace('jpg', 'xml')
        xml_path = os.path.join('./converted_xml', xml_name)
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        id_list = []
        cls_list  = []
        rela_list = []
        box_list  = []
        for item_num in range(xml_root):
            xml_item = xml_root[item_num]
            assert xml_item[-1].tag == 'relation'

            if xml_item.tag == 'size':
                width = float(xml_item[0].text)
                height = float(xml_item[1].text)
                ratio = 1024.0 / max(width, height)

            if xml_item.tag == 'object':
                obj_name = xml_item[0].text
                obj_id = xml_item[1].text
                xmin = float(int(item[3][0].text))*ratio
                ymin = float(int(item[3][1].text))*ratio
                xmax = float(int(item[3][2].text))*ratio
                ymax = float(int(item[3][3].text))*ratio
                obj_box = [int(xmin), int(ymin), int(xmax), int(ymax)]

                id_list.append(obj_id)
                cls_list.append(ind_to_classes.index(obj_name))
                box_list.append(obj_box)

            if xml_item.tag == 'relation':
                subject_id = xml_item[0].text
                object_id = xml_item[1].text
                rela = xml_item[2].text

                rela_triplet = [id_list.index(subject_id),
                                id_list.index(object_id),
                                ind_to_predicates.index(rela)]

                rela_list.append(rela_triplet)
        cls_gather.append(np.array(cls_list))
        rela_gather.append(np.array(rela_list))
        box_gather.append(np.array(box_list))
    return split_mask, box_gather, cls_gather, rela_gather



def load_hrvg_info():
    load_dict = np.load('./vocab_ind.npy')
    ind_to_classes = load_dict.item().get('obj_ind')
    ind_to_predicates = load_dict.item().get('rela_ind')
    return ind_to_classes, ind_to_predicates


