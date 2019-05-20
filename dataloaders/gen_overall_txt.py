import os, sys
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import torch

from tqdm import tqdm

import pdb

def gen_txt(raw_folder, file_list):
    write_file = open('triplet_record.txt', 'a')
    count_bar = tqdm(total=len(file_list))
    img_counter = 0
    for file_name in file_list:
        count_bar.update(1)
        xml_path = os.path.join(raw_folder, file_name)
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        object_dict = {}
        total_xml_num = len(xml_root)
        if xml_root[total_xml_num-1].tag != 'relation':
            continue
        else:
            img_counter += 1
        for item_num in range(total_xml_num):
            item = xml_root[item_num]
            if item.tag == 'object':
                assert item[0].tag == 'name'
                assert item[1].tag == 'object_id'
                object_dict[item[1].text] = item[0].text
                continue

            if item.tag == 'relation':
                assert item[2].tag == 'predicate'
                content = ','.join([object_dict[item[0].text],
                                    object_dict[item[1].text],
                                    item[2].text])
                write_file.write(content+'\n')
    write_file.close()
    count_bar.close()
    print('used image: %d' % img_counter)

if __name__ == '__main__':
    folder_path = './converted_xml'
    file_list = os.listdir(folder_path)
    gen_txt(folder_path, file_list)
