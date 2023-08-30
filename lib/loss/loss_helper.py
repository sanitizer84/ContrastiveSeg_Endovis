##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import os
import torch
import torch.nn as nn
# import numpy as np
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.loss.rmi_loss import RMILoss
from lib.loss.lovasz_loss import lovasz_softmax_flat, flatten_probas


class FSCELOVASZLoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELOVASZLoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()
            Log.info('{} weights used.'.format(weight))

        reduction = 'mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        if isinstance(inputs, dict) and 'seg' in inputs:
            inputs = inputs['seg']
        loss = 0.0
        target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
        loss = self.ce_loss(inputs, target)
        pred = F.softmax(input=inputs, dim=1)
        loss_lovasz = lovasz_softmax_flat(*flatten_probas(pred, target, self.ignore_index),
                                            only_present=True)
        loss += loss_lovasz*1.5
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        # duhj
        if (targets_.size(1), targets_.size(2)) == scaled_size:
            return targets_
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in (params := self.configer.get('loss', 'params')):
            # weight = self.configer.get('loss', 'params')['ce_weight']
            weight = params['ce_weight']
            weight = torch.FloatTensor(weight).cuda()
            Log.info('custom weights used.')
        reduction = 'mean'
        
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']
        ignore_index = -1
        
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)


    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        # duhj
        if (targets_.size(1), targets_.size(2)) == scaled_size:
            return targets_
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSCERMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCERMILoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        if isinstance(inputs, dict) and 'seg' in inputs:
            inputs = inputs['seg']
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

            loss_rmi = self.rmi_loss(inputs, target)

            loss = loss + loss_rmi

        return loss

class FSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxRMILoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss += self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss
    

class SegFixLoss(nn.Module):
    """
    We predict a binary mask to categorize the boundary pixels as class 1 and otherwise as class 0
    Based on the pixels predicted as 1 within the binary mask, we further predict the direction for these
    pixels.
    """

    def __init__(self, configer=None):
        super().__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def calc_weights(self, label_map, num_classes):

        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()

    def forward(self, inputs, targets, **kwargs):

        from lib.utils.helpers.offset_helper import DTOffsetHelper

        pred_mask, pred_direction = inputs

        seg_label_map, distance_map, angle_map = targets[0], targets[1], targets[2]
        gt_mask = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map, return_tensor=True)

        gt_size = gt_mask.shape[1:]
        mask_weights = self.calc_weights(gt_mask, 2)

        pred_direction = F.interpolate(pred_direction, size=gt_size, mode="bilinear", align_corners=True)
        pred_mask = F.interpolate(pred_mask, size=gt_size, mode="bilinear", align_corners=True)
        mask_loss = F.cross_entropy(pred_mask, gt_mask, weight=mask_weights, ignore_index=-1)

        mask_threshold = float(os.environ.get('mask_threshold', 0.5))
        binary_pred_mask = torch.softmax(pred_mask, dim=1)[:, 1, :, :] > mask_threshold

        gt_direction = DTOffsetHelper.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label_map,
            extra_ignore_mask=(binary_pred_mask == 0),
            return_tensor=True
        )

        direction_loss_mask = gt_direction != -1
        direction_weights = self.calc_weights(gt_direction[direction_loss_mask], pred_direction.size(1))
        direction_loss = F.cross_entropy(pred_direction, gt_direction, weight=direction_weights, ignore_index=-1)

        if self.training \
                and self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 \
                and torch.cuda.current_device() == 0:
            Log.info('mask loss: {} direction loss: {}.'.format(mask_loss, direction_loss))

        mask_weight = float(os.environ.get('mask_weight', 1))
        direction_weight = float(os.environ.get('direction_weight', 1))

        return mask_weight * mask_loss + direction_weight * direction_loss

class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss