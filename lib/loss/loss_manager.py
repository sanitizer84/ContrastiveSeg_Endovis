##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib.loss.loss_helper import FSCELoss, FSAuxCELoss, FSCELOVASZLoss, FSAuxRMILoss, FSCERMILoss
from lib.loss.loss_helper import SegFixLoss
from lib.loss.loss_contrast import ContrastCELoss
from lib.loss.loss_contrast_mem import MemContrastCELoss
from lib.loss.rmi_loss import RMILoss
from lib.utils.tools.logger import Logger as Log


SEG_LOSS_DICT = {
    'contrast_ce_loss':     ContrastCELoss,
    'contrast_ce_loss_mem': MemContrastCELoss,
    'fs_ce_loss':           FSCELoss,
    'segfix_loss':          SegFixLoss,
    'fs_ce_lovasz_loss':    FSCELOVASZLoss,
    'fs_auxce_loss':        FSAuxCELoss,
    'fs_ce_rmi_loss':       FSCERMILoss,
    'fs_aux_rmi_loss':      FSAuxRMILoss,
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        # if is_distributed():
        Log.info('use distributed loss')
        return loss
            

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key in SEG_LOSS_DICT:
            Log.info('use loss: {}.'.format(key))
            loss = SEG_LOSS_DICT[key](self.configer)
            return self._parallel(loss)
        Log.error('Loss: {} not valid!'.format(key))
        exit(1)
        
        
        


