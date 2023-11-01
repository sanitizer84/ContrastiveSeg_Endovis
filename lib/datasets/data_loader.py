##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret, JingyiXie
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import torch
from torch.utils import data

import lib.datasets.tools.transforms as trans
from lib.datasets.tools import cv2_aug_transforms
from lib.datasets.loader.default_loader import DefaultLoader, CSDataTestLoader
from lib.datasets.tools.collate import collate
from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_world_size, get_rank, is_distributed


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        
        self.aug_train_transform = cv2_aug_transforms.CV2AugCompose(self.configer, split='train')
        self.aug_val_transform = cv2_aug_transforms.CV2AugCompose(self.configer, split='val')

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])

    def get_dataloader_sampler(self, klass, split, dataset):

        root_dir = self.configer.get('data', 'data_dir')
        if isinstance(root_dir, list) and len(root_dir) == 1:
            root_dir = root_dir[0]

        kwargs = dict(
            dataset=dataset,
            aug_transform=(self.aug_train_transform if split == 'train' else self.aug_val_transform),
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            configer=self.configer
        )

        if isinstance(root_dir, str):
            loader = klass(root_dir, **kwargs)
        else:
            raise RuntimeError('Unknown root dir {}'.format(root_dir))
        if is_distributed():
            sampler = torch.utils.data.distributed.DistributedSampler(loader)
        else:
            sampler = None


        return loader, sampler

    def get_trainloader(self):
        klass = DefaultLoader    
        loader, sampler = self.get_dataloader_sampler(klass, 'train', 'train')
        trainloader = data.DataLoader(
            loader,
            batch_size=self.configer.get('train', 'batch_size') // get_world_size(), 
            pin_memory=True,
            num_workers=self.configer.get('data', 'workers') // get_world_size(),
            sampler=sampler,
            shuffle=(sampler is None),
            prefetch_factor=4,  #duhj
            persistent_workers=True,    #duhj
            drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('train', 'data_transformer')
            )
        )
        return trainloader

    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset
        if self.configer.get('method') == 'fcn_segmentor':
            #default manner: load the ground-truth label.
            if get_rank() == 0: Log.info('use DefaultLoader for {}:'.format(dataset))
            klass = DefaultLoader
        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

        loader, sampler = self.get_dataloader_sampler(klass, 'val', dataset)
        valloader = data.DataLoader(
            loader,
            sampler=sampler,
            batch_size=self.configer.get('val', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('val', 'data_transformer')
            )
        )
        return valloader

    def get_testloader(self, dataset=None):
        dataset = 'test' if dataset is None else dataset
        if self.configer.get('method') == 'fcn_segmentor':
            Log.info('use CSDataTestLoader for test ...')
            root_dir = self.configer.get('data', 'data_dir')
            if isinstance(root_dir, list) and len(root_dir) == 1:
                root_dir = root_dir[0]
            test_loader = data.DataLoader(
                CSDataTestLoader(root_dir=root_dir, dataset=dataset,
                                 img_transform=self.img_transform,
                                 configer=self.configer),
                batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'), shuffle=False,
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('test', 'data_transformer')
                )
            )
            return test_loader


if __name__ == "__main__":
    pass
