## 没有timm库，需要pip install timm

lib/datasets/tools/cv2_aug_transforms.py 519行， collections.Iterable 改为 collections.abc.Iterable

建pretrained_model目录，下载存放hrnetv2_w48_imagenet_pretrained.pth

dataset/preprocess/coco_stuff_generator里生成的路径里改成image不是images

loss_contrast.py  174行， forward函数一开始增加以下内容。preds多了一层列表，尚不知原因，这样可以运行。
        # duhj
        preds = preds[0]
        # duhj  



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
## Assertion t >= 0 && t < n_classes failed.错误。 这种情况发生在deeplabv3主干模型
## deeplabv3模型分割，R_101_D_8.json文件中，分割的类别数量和类别序号，必须是从0开始的，连续的序列，否则会报下面的错误！
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



batch 24
   0.4986675312653307
[base.py, 49] Class IOU: {0: 0.8704773653730417, 1: 0.4082939211274972, 2: 0.5059247196607044, 3: 0.9351050765422761, 4: 0.4769584480800104, 5: 0.3558847382026459, 6: 0.0009498480243161094, 7: 0.49692499205046636, 8: 0.4374886723270177}
   0.4880928258927748
[base.py, 49] Class IOU: {0: 0.8709527475632857, 1: 0.39583528342571805, 2: 0.5474965222458377, 3: 0.9208473652219391, 4: 0.5009193007765507, 5: 0.28157782590674785, 6: 0.0023964032743775715, 7: 0.4735072951339639, 8: 0.39930268948655256}






重新调整了数据集  测试集 dataset1,9个类, batchsize 8
ce noweight 
--- 0.39896887000407083
[base.py, 49] Class IOU: {0: 0.7982652083029121, 1: 0.2916431862522365, 2: 0.3982303880334316, 3: 0.9081195361653865, 4: 0.14571687801187874, 5: 0.3007271815446339, 6: 0.0, 7: 0.4434549155103987, 8: 0.3045625362157595}

原始对比学习
--- 0.3801999172768762
[base.py, 49] Class IOU: {0: 0.8245693536565765, 1: 0.3378764619789684, 2: 0.3113370202703216, 3: 0.9195357107638112, 4: 0.1906581162859191, 5: 0.22760429677741695, 6: 0.0, 7: 0.314400896854328, 8: 0.2958173989045437}

ce + 原对比学习
--- 0.41290537182318654
[base.py, 49] Class IOU: {0: 0.7854592867343667, 1: 0.36407444720141574, 2: 0.3386346380575917, 3: 0.9071971626203046, 4: 0.2733249633165933, 5: 0.2756971450099934, 6: 0.0, 7: 0.372589674201422, 8: 0.39917102926699116}



### 改策略后： lr 0.008 batch 8  ce+自己的对比学习 44700iter之内

ce*0.5 :
0.5823342036698262
[base.py, 49] Class IOU: {0: 0.9092338732874209, 1: 0.520638444697743, 2: 0.6698655295815457, 3: 0.9580051480526014, 4: 0.5651273687192698, 5: 0.5528214389856176, 6: 0.0023762544289563574, 7: 0.6149668686295569, 8: 0.4479729066457249}


### "temperature": 0.12   [-2000:-1]   "base_temperature": 0.07,      "max_samples": 7000,
 0.5986699644428407
[base.py, 49] Class IOU: {0: 0.9011651664541309, 1: 0.5107483534380257, 2: 0.6992526513890494, 3: 0.9553094109136516, 4: 0.5426854549223702, 5: 0.7153776489766347, 6: 0.0015272033089405027, 7: 0.566655976798338, 8: 0.4953078137844248}

 0.6103943140712494
[base.py, 49] Class IOU: {0: 0.9118148503483745, 1: 0.5278684624284982, 2: 0.6687970594709424, 3: 0.962942580022808, 4: 0.5890172124448055, 5: 0.6550990159695987, 6: 0.0, 7: 0.6699749432307572, 8: 0.5080347027254601}


只有对比学习 
 0.019152729332631298
[base.py, 49] Class IOU: {0: 0.0036443466360367213, 1: 0.015244183044226539, 2: 0.012411022297231643, 3: 0.10701389955503861, 4: 0.013029479861151248, 5: 0.0011451346030462303, 6: 0.0019946279816655953, 7: 0.003027038413850189, 8: 0.014864831601434921}

----------------------------------------------------------------------------------------------------------------
## deeplabv3: 因为resnet 101 主干要求类别必须从0开始连续编号，所以下面的0为背景，统计数据时要去掉
fs_ce: 0.39900525874295645
[base.py, 49] Class IOU: {0: 0.8136051922309636, 1: 0.7750811581770737, 2: 0.33046958464777815, 3: 0.4672793159156087, 4: 0.8244018927066094, 5: 0.04482046879377752, 6: 0.2591514575506022, 7: 0.0, 8: 0.21925330310967528, 9: 0.2559902142974759}

ce_aux_contrastive: bth:8  lr:0.008,  ce loss*1

ce + old contrastive
 0.3977564049672402
[base.py, 49] Class IOU: {0: 0.7104248997494401, 1: 0.7911564636199903, 2: 0.4350652579241765, 3: 0.48602449295970696, 4: 0.6306569759113831, 5: 0.04419710425961888, 6: 0.42007174996959745, 7: 0.0, 8: 0.16871875222140606, 9: 0.29124835305708213}

ce + my contrastive

 0.4941301846711056
[base.py, 49] Class IOU: {0: 0.8298780377356052, 1: 0.8100316146453361, 2: 0.4808694043424956, 3: 0.6349569684591773, 4: 0.8524483085938662, 5: 0.13559280736806198, 6: 0.5466768083647583, 7: 0.0, 8: 0.28251450121370786, 9: 0.36833339598804776}





endovis 2017数据集，做的是instruments seg,对应用的是instruments masks
### 测试数据集  各dataset的类别：
dataset1:0,1,2          计算结果的类：1,2
dataset2:0,2                        2
dataset3:0,3                        3
dataset4:0,2,3                      2,3
dataset5:0,1,4                      1,4
dataset6:0,2,3                      2,3
dataset7:0,1,4                      1,4
dataset8:0,1,6                      1,6
dataset9:0,1,2,6                    1,2,6
dataset10:0,3                       3


