##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, JingyiXie, LangHuang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank   #, is_distributed

class Trainer(object):
    # The class for Pose Estimation. Include train, val, val & predict.
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        # self.foward_time = AverageMeter()
        # self.backward_time = AverageMeter()
        # self.loss_time = AverageMeter()
        # self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        
        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()

        # try:
        #     from mmcv.cnn import get_model_complexity_info
        #     flops, params = get_model_complexity_info(self.seg_net, (3, 512, 512))
        #     split_line = '=' * 30
        #     print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        #         split_line, (3, 512, 512), flops, params))
        #     print('Be cautious if you use the results in papers. '
        #           'You may need to check if all ops are supported and verify that the '
        #           'flops computation is correct.')
        # except:
        #     pass

        self.seg_net = self.module_runner.load_net(self.seg_net)

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()
        self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' in key:
                bb_lr.append(value)
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            else:
                nbb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self):
        # Train function of every epoch during train phase.
        cudnn.enabled = True
        cudnn.benchmark = True
        world_size = get_world_size()
        
        def reduce_tensor(inp):
            # Reduce the loss from all processes so that process with rank 0 has the averaged results.
            if world_size > 1:
                with torch.no_grad():
                    dist.reduce(inp, dst=0)
                return inp
        
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()
        # scaler = torch.cuda.amp.GradScaler()

        # start training
        for i, data_dict in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(
                    self.configer.get('iters'),
                    self.scheduler, self.optimizer, backbone_list=[0, ]
                )
                
            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
            # with torch.cuda.amp.autocast():
            outputs = self.seg_net(*inputs)
            loss = self.pixel_loss(outputs, targets)
            backward_loss = loss
            display_loss = reduce_tensor(backward_loss) / world_size

            self.train_losses.update(display_loss.item(), batch_size)
            # scaler.scale(backward_loss).backward()
            # scaler.step(self.optimizer)
            # scaler.update()
            
            self.optimizer.zero_grad()
            backward_loss.backward()
            self.optimizer.step()
            
            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:

                Log.info('Iter:{0}, Loss:{1}, LR:{2}, Time:{batch_time.sum:.1f}s'.format(
                    self.configer.get('iters'),                   
                    self.train_losses.val,
                    self.module_runner.get_lr(self.optimizer),
                    batch_time=self.batch_time
                    ))
 
                self.batch_time.reset()
                self.train_losses.reset()
                
            # Check to val the current model.
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()
  
            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break
        
        

    def __val(self, data_loader=None):
        # Validation function during the train phase.
        cudnn.benchmark = True
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            self.optimizer.zero_grad()
            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            with torch.no_grad():
                outputs = self.seg_net(*inputs)
                loss = self.pixel_loss(
                    outputs, targets,
                    gathered=self.configer.get('network', 'gathered')
                )

                self.val_losses.update(loss.item(), batch_size)
                if isinstance(outputs, dict):
                    try:
                        outputs = outputs['pred']
                    except:
                        outputs = outputs['seg']
                self.evaluator.update_score(outputs, data_dict['meta'])

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.evaluator.update_performance()

        self.configer.update(['val_loss'], self.val_losses.avg)
        self.module_runner.save_net(self.seg_net, save_mode='performance')
        # self.module_runner.save_net(self.seg_net, save_mode='val_loss')
        
        # Print the log info & reset the states.
        self.evaluator.reduce_scores()
        if get_rank() == 0:
            Log.info('Test Time {t.sum:.3f}s, Loss {l.avg:.8f}'.format(t=self.batch_time, l=self.val_losses))
            self.evaluator.print_scores()

        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()
        
        self.seg_net.train()
        self.pixel_loss.train()

    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
                return
            elif self.configer.get('network', 'resume_train'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
                return
            # return

        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
            return

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train()

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))


if __name__ == "__main__":
    pass
