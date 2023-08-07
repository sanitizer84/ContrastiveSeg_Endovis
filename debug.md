没有timm库，需要pip install timm

lib/datasets/tools/cv2_aug_transforms.py 519行， collections.Iterable 改为 collections.abc.Iterable

建pretrained_model目录，下载存放hrnetv2_w48_imagenet_pretrained.pth

dataset/preprocess/coco_stuff_generator里生成的路径里改成image不是images

loss_contrast.py  174行， forward函数一开始增加以下内容。preds多了一层列表，尚不知原因，这样可以运行。
        # duhj
        preds = preds[0]
        # duhj  