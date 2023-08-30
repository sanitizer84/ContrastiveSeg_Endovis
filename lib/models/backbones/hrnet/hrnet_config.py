# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Rainbowsecret (yuyua@microsoft.com)
# ------------------------------------------------------------------------------

from yacs.config import CfgNode as CN

# configs for HRNet48
HRNET_48 = CN()
HRNET_48.STEM_INPLANES = 64
HRNET_48.FINAL_CONV_KERNEL = 1
HRNET_48.WITH_HEAD = True

HRNET_48.STAGE2 = CN()
HRNET_48.STAGE2.NUM_MODULES = 1
HRNET_48.STAGE2.NUM_BRANCHES = 2
HRNET_48.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_48.STAGE2.NUM_CHANNELS = [48, 96]
HRNET_48.STAGE2.BLOCK = 'BASIC'
HRNET_48.STAGE2.FUSE_METHOD = 'SUM'

HRNET_48.STAGE3 = CN()
HRNET_48.STAGE3.NUM_MODULES = 4
HRNET_48.STAGE3.NUM_BRANCHES = 3
HRNET_48.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_48.STAGE3.NUM_CHANNELS = [48, 96, 192]
HRNET_48.STAGE3.BLOCK = 'BASIC'
HRNET_48.STAGE3.FUSE_METHOD = 'SUM'

HRNET_48.STAGE4 = CN()
HRNET_48.STAGE4.NUM_MODULES = 3
HRNET_48.STAGE4.NUM_BRANCHES = 4
HRNET_48.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_48.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
HRNET_48.STAGE4.BLOCK = 'BASIC'
HRNET_48.STAGE4.FUSE_METHOD = 'SUM'

MODEL_CONFIGS = {
    'hrnet48': HRNET_48,
}
