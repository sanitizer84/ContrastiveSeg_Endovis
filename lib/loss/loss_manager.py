##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib.loss.loss_helper import FSRMILoss, FSCELoss, FSAuxCELoss, FSAuxRMILoss, FSCELOVASZLoss, MSFSAuxRMILoss
from lib.loss.loss_helper import SegFixLoss
from lib.loss.rmi_loss import RMILoss
from lib.loss.loss_contrast import ContrastAuxCELoss, ContrastCELoss
from lib.loss.loss_contrast_mem import ContrastCELoss as MemContrastCELoss

from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import is_distributed


SEG_LOSS_DICT = {
    'contrast_auxce_loss':  ContrastAuxCELoss,
    'contrast_ce_loss':     ContrastCELoss,
    'mem_contrast_ce_loss': MemContrastCELoss,
    
    'fs_ce_loss':           FSCELoss,
    'fs_auxce_loss':        FSAuxCELoss,
    'fs_aux_rmi_loss':      FSAuxRMILoss,
    'segfix_loss':          SegFixLoss,
    'rmi_loss':             RMILoss,
    'fs_rmi_loss':          FSRMILoss,
    'fs_ce_lovasz_loss':    FSCELOVASZLoss,
    'ms_fs_aux_rmi_loss':   MSFSAuxRMILoss,
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if is_distributed():
            Log.info('use distributed loss')
            return loss
            
        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            Log.info('use DataParallelCriterion loss')
            from lib.extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key in SEG_LOSS_DICT:
            Log.info('use loss: {}.'.format(key))
            loss = SEG_LOSS_DICT[key](self.configer)
            return self._parallel(loss)
        Log.error('Loss: {} not valid!'.format(key))
        exit(1)
        
        
        


