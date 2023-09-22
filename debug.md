## 没有timm库，需要pip install timm

lib/datasets/tools/cv2_aug_transforms.py 519行， collections.Iterable 改为 collections.abc.Iterable

建pretrained_model目录，下载存放hrnetv2_w48_imagenet_pretrained.pth

dataset/preprocess/coco_stuff_generator里生成的路径里改成image不是images

loss_contrast.py  174行， forward函数一开始增加以下内容。preds多了一层列表，尚不知原因，这样可以运行。
        # duhj
        preds = preds[0]
        # duhj  


## Token-to-Token ViT
论文：Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet

论文链接：https://arxiv.org/pdf/2101.11986.pdf

论文解读： https://zhuanlan.zhihu.com/p/354522966

论文解读：https://zhuanlan.zhihu.com/p/465148038

摘要：ViT在没有庞大数据规模的数据集上效果不如传统的ResNet。T2T-ViT相比于ViT，参数量和MACs(Multi-Adds)减少了200%，性能在ImageNet上有2.5%的提升，又快又强。


## gitee提交代码步骤：
  683  git add .
  684  git commit -m "deeplabV3原版和对比学习都可以运行"
  685  git push -f  contrastive-seg master



datasets/preprocess下各数据集的generator.py，都是将数据集中的彩色掩码图片，转换成单通道的掩码。这一工作完成之后，才能进行训练。


## 图片分辨率256会报错，估计是不能低于512

## 配置文件json中， label_class_ids原名label_list，容易和代码中的label文件名列表混淆，改名。这个列表内容应该是label图像中，

## label灰度图的预处理
最好自己提前做好label图预处理，这样训练过程中不必反复动态生成多个类别的灰度label。背景用255！！
需要分割的类别在灰度图像中的数值置为0-254范围内平均间隔的数字，如[0, 50, 100]， 在训练的配置json文件的data:label_class_ids列表中也要对应一致。
训练集和验证集的图，应该是训练集的每个小类别（可能不同小类别之间类别出现或分布特征有差异）都在验证集有对应的部分图片，结果才会好。
损失函数的weight可以设置。应统计数据集中各类的比例，类别间差距较大（>5倍？），则应计算各类在损失函数计算中的weight，在配置json的data:ce_weight处指明。



## 自己的数据集：
first of all, you need to create a set of config files under the folder openseg.pytorch/configs/your_dataset_name following the other dataset. For example, we take the coco_stuff dataset as an example (as below),
openseg.pytorch/configs/coco_stuff/R_101_D_8.json

Lines 2 to 49 in db0d389

  "dataset": "coco_stuff", 
  "method": "fcn_segmentor", 
  "data": { 
    "image_tool": "cv2", 
    "input_mode": "BGR", 
    "num_classes": 171, 
    "label_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,  
                  21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,  
                  40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,  
                  59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,  
                  78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96,  
                  97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,  
                  113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,  
                  129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,  
                  145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,  
                  161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,  
                  177, 178, 179, 180, 181, 182], 
    "reduce_zero_label": true, 
    "data_dir": "~/DataSet/pascal_context", 
    "workers": 8 
  }, 
 "train": { 
    "batch_size": 16, 
    "data_transformer": { 
      "size_mode": "fix_size", 
      "input_size": [520, 520], 
      "align_method": "only_pad", 
      "pad_mode": "random" 
    } 
  }, 
  "val": { 
    "batch_size": 4, 
    "mode": "ss_test", 
    "data_transformer": { 
      "size_mode": "diverse_size", 
      "align_method": "only_pad", 
      "pad_mode": "pad_right_down" 
    } 
  }, 
  "test": { 
    "mode": "ss_test", 
    "batch_size": 4, 
    "crop_size": [520, 520], 
    "scale_search": [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], 
    "data_transformer": { 
      "size_mode": "diverse_size" 
    } 
  }, 
You need to change a set of keywords in the json file including the "dataset", "num_classes", "label_list", "reduce_zero_label", "input_size","crop_size", "base_lr" and so on. Of course, you can also reset these parameters in the training script file (listed as below),

openseg.pytorch/scripts/coco_stuff/run_h_48_d_4_ocr_train.sh

Lines 31 to 48 in db0d389

 if [ "$1"x == "train"x ]; then 
   ${PYTHON} -u main.py --configs ${CONFIGS} \ 
                        --drop_last y \ 
                        --nbb_mult 10 \ 
                        --phase train \ 
                        --gathered n \ 
                        --loss_balance y \ 
                        --log_to_file n \ 
                        --backbone ${BACKBONE} \ 
                        --model_name ${MODEL_NAME} \ 
                        --gpu 0 1 2 3 \ 
                        --data_dir ${DATA_DIR} \ 
                        --loss_type ${LOSS_TYPE} \ 
                        --max_iters ${MAX_ITERS} \ 
                        --checkpoints_name ${CHECKPOINTS_NAME} \ 
                        --pretrained ${PRETRAINED_MODEL} \ 
                        2>&1 | tee ${LOG_FILE} 
                         
second, you need to organize your training/validation dataset following the folder structure like below,
├── your_dataset_name
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
third, you need to prepare the training script following the example below and change the DATA_DIR, SAVE_DIR, CONFIGS, and all of the other settings accordingly.
https://github.com/openseg-

标签类别数量不平衡，max/min > 10 要考虑交叉熵损失的weights自定义
Pytorch的nn.CrossEntropyLoss()的weight怎么使用？
分割实验，label标注的0-3四类，0类的比重过大，1类其次，2，3类都很少，怎么使用loss的weight来减轻样本不平衡问题？

如何设置weight才能提升分类的性能。

一般情况下，假设 
 表示数量最多类别的样本个数，
表示数量最少类别的样本个数，当 
 的时候，是不需要考虑样本不平衡问题的。当它们的比值大于（或者远远）10的时候是要考虑样本不平衡问题，为什么要考虑样本不平衡问题呢？接下来我们来解释一下：

假设有三类，标签类别为0, 1, 2，所对应的样本数量为100000，100， 10。此时就有一个问题，在网络学习的过程中，假设预测出来的标签都是0（100000个样本），它的准确率为 
 ，将近100%，所以模型就会朝着拟合标签0的方向更新，导致对标签0的样本过拟合，对其它两个类别的样本欠拟合，泛化能力很差。

那我们来解释一下，nn.CrossEntropyLoss()的weight如何解决样本不平衡问题的。

当类别中的样本数量不均衡的时候，对于训练图像数量较少的类，你给它更多的权重，这样如果网络在预测这些类的标签时出错，就会受到更多的惩罚。对于具有大量图像的类，您可以赋予它较小的权重。

那我们如何选择weight的值呢？一般有两种方式（建议第二种）：

第一种，用样本数的倒数当做权重。即 
 。用上述的三分类表示就是， 
 。之前测试过几次，这种方式会造成分类准确率下降。
第二种，Max(Numberof occurrences in most common class) / (Number of occurrences in rare classes)。即用类别中最大样本数量除以当前类别样本的数量，作为权重系数。 用上述的三分类表示就是， 
 。代码表示如下：
weights = [1.0, 1000, 10000]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

## label的多种类别如果适用deeplab等模型，pytorch默认是连续编号的，1，2，3...
## Assertion t >= 0 && t < n_classes failed.错误。
最近在训练图像4分类时，为了使标签可视化，分别用了不同的像素值[0, 74, 128, 145]对图像做了mask像素分割。
在模型训练时遇到如下两个bug：
1.RuntimeError: CUDA error: device-side assert triggered CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
2.[00:00<?, ?img/s]C:\actions-runner_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:103: block: [0,0,0], thread: [832,0,0] Assertion t >= 0 && t < n_classes failed.
查找问题：
1.，mask对照了我自己的label种类发现都是没有问题的！(除了标签外，不能有其他多余的像素值在图中)
2，将标签图mask中像素的值为[0, 74, 128, 145]的pix，转换pix为[0,1,2,3]

为何要这样编码：
在图像分割任务中，标签从0开始的连续值是一种常见的表示方式。这种表示方式的目的是方便计算机进行处理和存储。
具体来说，标签是用来标识图像中不同的分类或语义区域，例如一幅图像中可能包含人、车、树等不同的物体或者地面、天空等不同的区域。在这个问题中，我们需要为每个物体或区域分配一个唯一的标签。如果我们用不同的数字给每个物体或区域编码，那么这些数字应该是连续的，并从0开始，如0表示背景，1表示人，2表示车等。
这种编码方式的好处在于，可以用一个数组或矩阵来表示整张图像，矩阵的每个元素存储一个数字，表示对应像素的标签。例如，在语义分割任务中，我们可以使用一个尺寸与原图像相同的矩阵，每个元素存储一个数字，表示对应像素所属的物体或区域。这样的编码方式对计算和存储都非常方便。
另外，从0开始的连续值的编码方式还可以避免混淆和错误。如果我们用非连续的或重复的数字来编码，可能会使结果不可解释或者出现错误。对于计算机来说，使用连续值的编码方式也更加直观和易于处理。




contrast json中的 proj_dim=256不能改，改了之后loss的维度会报错
deeplabV3的对比学习，带mem的部分没有实现代码



## 小规模测试，结果记录：
hrnet:
  ce no weight： 10000 iter,            MIOU:0.7379
  ce[1, 5, 10, 10], 10000 iter,         MIOU:0.7823
  ce[1, 5, 10, 10] + lovaz, 10000 iter, MIOU:0.7828
  lovaz 0.7729  0.7748
  contrast,                             zas 0.7886, lr=0.00969  (contrast weight=0.1)
                                        0.8088, lr=0.00066  weight=1
  contrast_mem:                         0.7832 lr=0.009329
  
  mem方式效果没有体现

deeplabv3:
  fs_auxce_loss:                         0.7173  10000iter还在继续升高，可能比较慢
  contrast_auxce_loss,                   0.7932

## 300张的结果：
ce * no weights     0.7979
ce * weights        0.7997
ce * contrast       0.8285
ce + contrast       0.8209   增大了max_samples 4096, max_view 2048 效果不好
ce + contrast       0.8380   大约是"max_samples": 1024, "max_views": 32,  max_views小一点，有助于样本数少的标签
ce + contrast       0.8404  再次验证了mem(默认参数)性能会降低，是否和数据集有关？    "max_views": 16。如果max_view大一些，不足的像素点重复填充，性能是否会提高？
ce*weight*0.01,+ contrast MIOU 0.6890
 num_hard > 0 and num_easy == 0:全部用hard_indices填充， 0.7332
 两种为0的都要，0.7525
                     0.7627  iter=9000
自己构造的对比样本：
0.7751  iter=14700, lamda_poly=1, 0.0002;
0.7718  iter=14700, lambda_poly=0.5, 
0.8198  三种情况都收集难易两类样本， max samples:6100, samples list:1048576, poly=0.9,但是出结果是lr已经为0
0.8135  max samples:4096,其他同上
0.8142  max samples:2048,其他同上
0.8208  samples list:1048576, poly 0.6, temperature 0.4, "max_samples": 6000, iter 6300
0.8217  samples list:1048576, poly 0.6, temperature 0.2, "max_samples": 6000, iter 6300
0.8276  上述参数，ce带权重，参数0.1
0.8350  上述参数，ce带权重，参数0.5
0.8356  上述参数，ce带权重，参数1







分类任务正样本选1，负样本尽可能多，可以拉大正负样本距离；分割中，不是这样，可能因为像素点与整张图不同
全部图片训练：2085张train ，其余val， 60000iter
ce 无权重：
---     0.85668， Class IOU: {0: 0.9347209540118968, 1: 0.9121641730140624, 2: 0.8225512421280061, 3: 0.8117808266466193, 4: 0.940838387542896, 5: 0.9309142639018967, 6: 0.6148148912247403, 7: 0.8624627515701646, 8: nan, 9: 0.7873295415610224, 10: 0.9279631305060331, 11: 0.866794198339732}
---    0.856938   Class IOU: {0: 0.9284695988331771, 1: 0.9116174095487221, 2: 0.8203813345414537, 3: 0.8187241871652117, 4: 0.9433562826782494, 5: 0.931469057488983, 6: 0.6191711613299652, 7: 0.858219035983495, 8: nan, 9: 0.7952660552497236, 10: 0.9024254037315799, 11: 0.897219507450312}
---    0.85669 
ce 12个类的权重：
[0.541718,
 0.91004837,
 0.9543705,
 0.9790279,
 0.82916045,
 0.8839496,
 0.9967673,
 0.99515355,
 0.99997556,
 0.99593496,
 0.9190495,
 0.99484426]
 --- 0.85227  Mean  IOU: 0.852275664162207
[base.py, 49] Class IOU: {0: 0.9253442526449319, 1: 0.9065465810576323, 2: 0.8189444310379784, 3: 0.8021501645686381, 4: 0.9391671472345614, 5: 0.9279011362262582, 6: 0.6065035101404056, 7: 0.8682383298146845, 8: nan, 9: 0.7711185143648778, 10: 0.9016599798778493, 11: 0.9074582588164608}
---  0.8537639762838828
[base.py, 49] Class IOU: {0: 0.9303020096351083, 1: 0.9085536652531635, 2: 0.8152188882925396, 3: 0.8116282808794566, 4: 0.9414307300201752, 5: 0.932710060560841, 6: 0.6120382951799106, 7: 0.8655127919497976, 8: nan, 9: 0.7519793138378695, 10: 0.9179550830672164, 11: 0.9040746204466313}





重新调整了数据集  测试集 dataset1,9个类
ce noweight 
--- 0.39896887000407083
[base.py, 49] Class IOU: {0: 0.7982652083029121, 1: 0.2916431862522365, 2: 0.3982303880334316, 3: 0.9081195361653865, 4: 0.14571687801187874, 5: 0.3007271815446339, 6: 0.0, 7: 0.4434549155103987, 8: 0.3045625362157595}

原始对比学习
--- 0.3801999172768762
[base.py, 49] Class IOU: {0: 0.8245693536565765, 1: 0.3378764619789684, 2: 0.3113370202703216, 3: 0.9195357107638112, 4: 0.1906581162859191, 5: 0.22760429677741695, 6: 0.0, 7: 0.314400896854328, 8: 0.2958173989045437}

ce + 原对比学习
--- 0.41290537182318654
[base.py, 49] Class IOU: {0: 0.7854592867343667, 1: 0.36407444720141574, 2: 0.3386346380575917, 3: 0.9071971626203046, 4: 0.2733249633165933, 5: 0.2756971450099934, 6: 0.0, 7: 0.372589674201422, 8: 0.39917102926699116}



ce + 自己的对比学习 
--- batch 4,  0.3778411428429629
[base.py, 49] Class IOU: {0: 0.7850337424081875, 1: 0.33423907932875496, 2: 0.27647190563631374, 3: 0.9083035448610712, 4: 0.1915135915135915, 5: 0.25093655823866, 6: 0.0, 7: 0.38406114657149765, 8: 0.2700107170285895}

--- batch 8   0.4553486444511129
[base.py, 49] Class IOU: {0: 0.8663513276489326, 1: 0.37836800734565434, 2: 0.468653944235495, 3: 0.9298326693487637, 4: 0.40277703066164605, 5: 0.28113793222189976, 6: 0.0, 7: 0.478668398697908, 8: 0.29234848989971657}

--- batch 24 0.4967137694763602, sample list:32768, temperature 0.2, max samples 6000
[base.py, 49] Class IOU: {0: 0.8694486033376373, 1: 0.4346639168177122, 2: 0.5218271814953002, 3: 0.9329730050175445, 4: 0.46681999102124555, 5: 0.4214280206646619, 6: 0.0, 7: 0.4587402024088946, 8: 0.36452300452424524}

