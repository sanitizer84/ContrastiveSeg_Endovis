import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.distributed import get_world_size, get_rank, is_distributed
from lib.utils.tools.average_meter import AverageMeter
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler


class Trainer(object):
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
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
        
        self.seg_net = self.module_runner.load_net(
            self.model_manager.semantic_segmentor()
        )

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader =   self.data_loader.get_valloader()
        self.pixel_loss =   self.loss_manager.get_seg_loss()
        self.pixel_loss =   self.module_runner.to_device(self.pixel_loss)

        if not self.configer.exists("contrast"):
            Log.info('contrast configure not found.')
            exit(1)
        

        # self.with_memory = self.configer.exists('contrast', 'with_memory')
        # if self.with_memory:
        #     self.memory_size = self.configer.get('contrast', 'memory_size')
        #     self.pixel_update_freq = self.configer.get('contrast', 'pixel_update_freq')

        self.network_stride = self.configer.get('network', 'stride')

        # Log.info("with_memory: {}".format(self.with_memory))

    # def _dequeue_and_enqueue(self, keys, labels,
    #                          segment_queue, segment_queue_ptr,
    #                          pixel_queue, pixel_queue_ptr):
    #     batch_size = keys.shape[0]
    #     feat_dim = keys.shape[1]

    #     labels = labels[:, ::self.network_stride, ::self.network_stride]

    #     for bs in range(batch_size):
    #         this_feat = keys[bs].contiguous().view(feat_dim, -1)
    #         this_label = labels[bs].contiguous().view(-1)
    #         this_label_ids = torch.unique(this_label)
    #         this_label_ids = [x for x in this_label_ids if x > 0]

    #         for lb in this_label_ids:
    #             idxs = (this_label == lb).nonzero()

    #             # segment enqueue and dequeue
    #             feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
    #             ptr = int(segment_queue_ptr[lb])
    #             segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
    #             segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.memory_size

    #             # pixel enqueue and dequeue
    #             num_pixel = idxs.shape[0]
    #             perm = torch.randperm(num_pixel)
    #             K = min(num_pixel, self.pixel_update_freq)
    #             feat = this_feat[:, perm[:K]]
    #             feat = torch.transpose(feat, 0, 1)
    #             ptr = int(pixel_queue_ptr[lb])

    #             if ptr + K >= self.memory_size:
    #                 pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
    #                 pixel_queue_ptr[lb] = 0
    #             else:
    #                 pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
    #                 pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % self.memory_size

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
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    # def __train(self):
    #     world_size = get_world_size()
    #     def reduce_tensor(inp):
    #         # Reduce the loss from all processes so that process with rank 0 has the averaged results.
    #         if world_size > 1:
    #             with torch.no_grad():
    #                 dist.reduce(inp, dst=0)
    #         return inp
        
    #     # Train function of every epoch during train phase.
    #     self.seg_net.train()
    #     self.pixel_loss.train()
    #     start_time = time.time()
    #     cudnn.benchmark = True

    #     for _, data_dict in enumerate(self.train_loader):                
    #         (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
    #         self.optimizer.zero_grad()
    #         outputs = self.seg_net(*inputs, with_embed=True)
    #         backward_loss = self.pixel_loss(outputs, targets, with_embed=True)
    #         backward_loss.backward()
    #         display_loss = reduce_tensor(backward_loss) / world_size
    #         self.train_losses.update(display_loss.item(), batch_size)

    #         self.optimizer.step()
    #         self.scheduler.step()

    #         # Update the vars of the train phase.
    #         self.batch_time.update(time.time() - start_time)
    #         start_time = time.time()
    #         self.configer.plus_one('iters')

    #         # Print the log info & reset the states.
    #         if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
    #             Log.info('Iter: {0} Time {batch_time.sum:.3f}s Lr = {1} Loss = {loss.val:.8f}'.format(
    #                 self.configer.get('iters'),
    #                 self.module_runner.get_lr(self.optimizer)[0],
    #                 batch_time=self.batch_time, 
    #                 loss=self.train_losses))
    #             self.batch_time.reset()
    #             self.train_losses.reset()

    #         if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
    #             self.__val()
    #         if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
    #             break


    def __val(self, data_loader=None):
        # Validation function during the train phase.
        self.seg_net.eval()
        self.pixel_loss.eval()

        # 为endovis2017数据集做的修改，共有10个dataset用于验证    
        dataset_list = ['val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'val8', 'val9', 'val10']
        for valid in dataset_list:
            data_loader = self.data_loader.get_valloader(valid)

            for j, data_dict in enumerate(data_loader):
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

                with torch.no_grad():

                    outputs = self.seg_net(*inputs, is_eval=True)
                    # loss = self.pixel_loss(outputs, targets)               
                    # self.val_losses.update(loss.item(), batch_size)
                    # print(outputs)
                    if isinstance(outputs, dict):
                        self.evaluator.update_score(outputs['seg'], data_dict['meta'])
                    else:
                        self.evaluator.update_score(outputs, data_dict['meta'])


            self.evaluator.update_performance()
            # self.configer.update(['val_loss'], self.val_losses.avg)

            # Print the log info & reset the states.
            if get_rank() == 0:
                Log.info(
                    'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.8f}\n'.format(
                        batch_time=self.batch_time, loss=self.val_losses))
                self.evaluator.print_scores()


    def train(self):
        # if self.configer.get('network', 'resume') is not None:
        #     if self.configer.get('network', 'resume_val'):
        #         self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        #         return
        #     elif self.configer.get('network', 'resume_train'):
        #         self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
        #         return
        #     # return

        # if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
        #     self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        #     return

        # while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
        #     self.__train()
        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))


if __name__ == "__main__":
    pass
