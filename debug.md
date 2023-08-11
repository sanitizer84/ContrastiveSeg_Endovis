没有timm库，需要pip install timm

lib/datasets/tools/cv2_aug_transforms.py 519行， collections.Iterable 改为 collections.abc.Iterable

建pretrained_model目录，下载存放hrnetv2_w48_imagenet_pretrained.pth

dataset/preprocess/coco_stuff_generator里生成的路径里改成image不是images

loss_contrast.py  174行， forward函数一开始增加以下内容。preds多了一层列表，尚不知原因，这样可以运行。
        # duhj
        preds = preds[0]
        # duhj  


Token-to-Token ViT
论文：Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet

论文链接：https://arxiv.org/pdf/2101.11986.pdf

论文解读： https://zhuanlan.zhihu.com/p/354522966

论文解读：https://zhuanlan.zhihu.com/p/465148038

摘要：ViT在没有庞大数据规模的数据集上效果不如传统的ResNet。T2T-ViT相比于ViT，参数量和MACs(Multi-Adds)减少了200%，性能在ImageNet上有2.5%的提升，又快又强。


gitee提交代码步骤：
  683  git add .
  684  git commit -m "deeplabV3原版和对比学习都可以运行"
  685  git push -f  contrastive-seg master
