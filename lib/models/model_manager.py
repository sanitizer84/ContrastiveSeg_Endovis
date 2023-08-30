##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
## Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# FCN baseline 
from lib.models.nets.fcnet import FcnNet

# HRNet
from lib.models.nets.hrnet import HRNet_W48, HRNet_W48_CONTRAST
from lib.models.nets.hrnet import HRNet_W48_OCR, HRNet_W48_OCR_B, HRNet_W48_OCR_CONTRAST, HRNet_W48_Contrast_MEM

# # SegFix
# from lib.models.nets.segfix import SegFix_HRNet

from lib.utils.tools.logger import Logger as Log
from lib.models.nets.deeplab import DeepLabV3, DeepLabV3Contrast

SEG_MODEL_DICT = {
    # HRNet series
    'hrnet_w48': HRNet_W48,
    'hrnet_w48_ocr': HRNet_W48_OCR,
    'hrnet_w48_ocr_b': HRNet_W48_OCR_B,
    # baseline series
    'fcnet': FcnNet,
    'hrnet_w48_contrast': HRNet_W48_CONTRAST,
    'hrnet_w48_ocr_contrast': HRNet_W48_OCR_CONTRAST,
    'hrnet_w48_contrast_mem': HRNet_W48_Contrast_MEM,
    'deeplab_v3': DeepLabV3,
    'deeplab_v3_contrast': DeepLabV3Contrast,
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')
        Log.error(model_name)
        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
