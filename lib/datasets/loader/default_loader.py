##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie, LangHuang, DonnyYou, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os


import numpy as np
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class DefaultLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.name_list = self.__list_dirs(root_dir, dataset)
        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'
        Log.info('{} {}'.format(root_dir, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        # Log.info('{}'.format(self.img_list[index]))
        img_size = ImageHelper.get_size(img)
        
        if self.configer.get('data', 'label_ready') == 'true':
            labelmap = ImageHelper.read_label(self.label_list[index],
                                            tool=self.configer.get('data', 'image_tool'), mode='P')
        else:
            
            # RGB模式读入label图，mode='P'返回nparray类型   duhj
            labelmap = ImageHelper.read_image(self.label_list[index],
                                            tool=self.configer.get('data', 'image_tool'), mode='P')
            # RGB模式读入label图，生成灰度图labelmap，用classid替换每类对应颜色，背景用255表示  duhj
            if self.configer.exists('data', 'label_class_ids'):
                labelmap = self._encode_label(labelmap)

        ori_target = ImageHelper.tonp(labelmap)
        
        # 背景在labelmap中用255表示，将其置为-1，后面在task.py的eval()不参与计算分数 duhj
        # ori_target[ori_target == 255] = -1
        ori_target[labelmap == 255] = -1

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap=labelmap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )
        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
        )

    def _reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_class_ids'))):
            class_id = self.configer.get('data', 'label_class_ids')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        # support the argument to pass the file list used for training/testing
        file_list_txt = os.environ.get('use_file_list')
        if file_list_txt is None:
            files = sorted(os.listdir(image_dir))
        else:
            Log.info("Using file list {} for training".format(file_list_txt))
            with open(os.path.join(root_dir, dataset, 'file_list', file_list_txt)) as f:
                files = [x.strip() for x in f]

        for file_name in files:
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}'.format(file_name))
            label_path = os.path.join(label_dir, image_name + '.bmp')
            # Log.info('{} {} {}'.format(image_name, img_path, label_path))
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} {} not exists.'.format(label_path, img_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            Log.info("Use validation dataset for training.")
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')          

            if file_list_txt is None:
                files = sorted(os.listdir(image_dir))
            else:
                Log.info("Using file list {} for validation".format(file_list_txt))
                with open(os.path.join(root_dir, 'val', 'file_list', file_list_txt)) as f:
                    files = [x.strip() for x in f]

            for file_name in files:
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}'.format(file_name))
                label_path = os.path.join(label_dir, image_name + '.png')
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} {} not exists.'.format(label_path, img_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)
               
        return img_list, label_list, name_list


class CSDataTestLoader(data.Dataset):
    def __init__(self, root_dir, dataset=None, img_transform=None, configer=None):
        self.configer = configer
        self.img_transform = img_transform
        self.img_list, self.name_list, self.subfolder_list = self.__list_dirs(root_dir, dataset)

        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = (size_mode != 'diverse_size')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        if self.img_transform is not None:
            img = self.img_transform(img)
        meta = dict(
            ori_img_size=img_size,
            border_size=img_size,
        )
        return dict(
            img=DataContainer(img, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
            subfolder=DataContainer(self.subfolder_list[index], stack=False, cpu_only=True),
        )

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        name_list = list()
        subfolder_list = list()
        image_dir = os.path.join(root_dir, dataset)
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        if self.configer.get('dataset') == 'cityscapes' or self.configer.get('dataset') == 'camvid':
            for item in os.listdir(image_dir):
                sub_image_dir = os.path.join(image_dir, item)
                for file_name in os.listdir(sub_image_dir):
                    image_name = file_name.split('.')[0]
                    img_path = os.path.join(sub_image_dir, file_name)
                    if not os.path.exists(img_path):
                        Log.error('Image Path: {} not exists.'.format(img_path))
                        continue
                    img_list.append(img_path)
                    name_list.append(image_name)
                    subfolder_list.append(item)
        else:
             for file_name in os.listdir(image_dir):
                image_name = file_name.split('.')[0]
                img_path = os.path.join(image_dir, file_name)
                if not os.path.exists(img_path):
                    Log.error('Image Path: {} not exists.'.format(img_path))
                    continue
                img_list.append(img_path)
                name_list.append(image_name)
                subfolder_list.append('')

        return img_list, name_list, subfolder_list

if __name__ == "__main__":
    # Test cityscapes loader.
    pass
