from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSCELoss, FSAuxCELoss
from lib.utils.tools.logger import Logger as Log
from main_contrastive import neg_list, pos_list

def add_samples(sample_list, id, newdata):
    if sample_list[id] is None:
        sample_list[id] = newdata
    else:
        sample_list[id] = torch.cat((sample_list[id], newdata), 0)[-4000:-1]

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')
        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']
        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = 32

    # my own contrastive seg
    def _hard_anchor_sampling(self, X, gt, y):
        global pos_list, neg_list
        # 这里gt是标签， y是预测值
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes = []
        total_classes = 0

        for ii in range(batch_size):
            this_classes = torch.unique(gt[ii]) #所有clasid
            classes.append([x for x in this_classes if x != self.ignore_label])     #去掉忽略的类
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples//total_classes
        easy_view = n_view // 2
        hard_view = n_view - easy_view
        
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        X_ptr = 0

        for ii in range(batch_size):
            # 批次里的每一张图
            this_gt = gt[ii]
            # this_y = y[ii]          
            for cls_id in classes[ii]:       #每张图中所有类别  
                id = int(cls_id)                  
                hard_indices = (this_gt != cls_id).nonzero()   
                easy_indices = (this_gt == cls_id).nonzero()      
                num_hard, num_easy = hard_indices.shape[0], easy_indices.shape[0] 
                if num_hard > 0 and num_easy > 0:
                    add_samples(neg_list, id, hard_indices) 
                    add_samples(pos_list, id, easy_indices) 
                    if 0 < num_hard < hard_view:                    # 构建当前错误样本序列  
                        hard_indices = hard_indices.repeat(hard_view//num_hard, 1)
                    else:
                        hard_indices = hard_indices[torch.randperm(num_hard)[:hard_view]]   
                            
                    if 0 < num_easy < easy_view:                    # 构建当前正确样本序列
                        easy_indices = easy_indices.repeat(easy_view//num_easy, 1)
                    else:
                        easy_indices = easy_indices[torch.randperm(num_easy)[:easy_view]]
                    
                elif num_hard > 0 and num_easy == 0:
                    add_samples(neg_list, id, hard_indices)     # 保存当前分类错误样本 
                    if num_hard < hard_view:                        # 构建当前错误样本序列
                        hard_indices = hard_indices.repeat(hard_view//num_hard, 1)
                    else:
                        hard_indices = hard_indices[torch.randperm(num_hard)[:hard_view]]     
                         
                    if pos_list[cls_id] != None:                    # 缺正确样本，从序列中取出填充
                        if 0 < pos_list[cls_id].size(0) < easy_view:
                            easy_indices = pos_list[cls_id].repeat(easy_view//pos_list[cls_id].size(0) , 1)
                        else:
                            easy_indices = pos_list[cls_id][torch.randperm(num_easy)[:easy_view]]    

                elif num_hard == 0 and num_easy >0:     
                    if pos_list[id] is not None: pos_list[id] = pos_list[id][:100]
                    add_samples(pos_list, id, easy_indices)     # 保存当前分类正确样本 
                    if 0 < num_easy < easy_view:                    # 构建当前正确样本序列
                        easy_indices = easy_indices.repeat(easy_view//num_easy, 1)
                    else:
                        easy_indices = easy_indices[torch.randperm(num_easy)[:easy_view]]
                        
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                X_[X_ptr, :indices.size(0), :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return X_, y_
    
    
  
    
    
    # # old contrastive of the paper Exploring....
    # def _hard_anchor_sampling(self, X, y_hat, y):
    #     # 这里y_hat是label， y是预测值
    #     batch_size, feat_dim = X.shape[0], X.shape[-1]

    #     classes = []
    #     total_classes = 0
    #     for ii in range(batch_size):
    #         this_y = y_hat[ii]
    #         this_classes = torch.unique(this_y) #所有clasid
    #         this_classes = [x for x in this_classes if x != self.ignore_label]  #去掉忽略的
    #         # ax_view是个门槛，低于此门槛的类不被计算！找出批次图像中所有数量>maxview个数的类别号clasid
    #         this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]   
    #         classes.append(this_classes)
    #         total_classes += len(this_classes)

    #     if total_classes == 0:
    #         return None, None

    #     n_view = min(self.max_samples//total_classes, self.max_views)

    #     X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
    #     y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

    #     X_ptr = 0
    #     for ii in range(batch_size):
    #         # 批次里的每一张图
    #         this_y_hat = y_hat[ii]
    #         this_y = y[ii]
    #         this_classes = classes[ii]
    #         #每个类别
    #         for cls_id in this_classes:
    #             # nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。
    #             # 它的返回值是一个长度为a.ndim(数组a的轴数)的元组，元组的每个元素都是一个整数数组，
    #             # 其值为非零元素的下标在对应轴上的值。
                
    #             #预测和label不一致
    #             hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()  
    #             #预测和label一致
    #             easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()  

    #             num_hard = hard_indices.shape[0]    #数量
    #             num_easy = easy_indices.shape[0]

    #             if num_hard >= n_view / 2 and num_easy >= n_view / 2:
    #                 # 都大于窗口尺寸，各为窗口一半
    #                 num_hard_keep = n_view // 2
    #                 num_easy_keep = n_view - num_hard_keep
    #             elif num_hard >= n_view / 2:
    #                 # easy的少
    #                 num_easy_keep = num_easy
    #                 num_hard_keep = n_view - num_easy_keep
    #             elif num_easy >= n_view / 2:
    #                 # hard的少
    #                 num_hard_keep = num_hard
    #                 num_easy_keep = n_view - num_hard_keep
    #             else:
    #                 Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
    #                 raise Exception
    #             # randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写
    #             # 此部分随机抽取要保留的那部分hard和easy像素点索引
    #             perm = torch.randperm(num_hard)
    #             # 打乱顺序后的前num_hard_keep个
    #             hard_indices = hard_indices[perm[:num_hard_keep]]
    #             perm = torch.randperm(num_easy)
    #             # 打乱顺序后的前num_easy_keep个
    #             easy_indices = easy_indices[perm[:num_easy_keep]]
    #             indices = torch.cat((hard_indices, easy_indices), dim=0)
    #             # 抽取出的像素赋值到空白矩阵作为返回值的
    #             X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
    #             y_[X_ptr] = cls_id
    #             X_ptr += 1
    #     # 此函数的目的是取出hard和easy的anchor，即像素点作为features X_, 标签cls_id作为y_
    #     return X_, y_
    
    


    def _contrastive(self, features_, labels_):
        anchor_num, n_view = features_.shape[0], features_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        # 转置后相等的坐标就是预测正确的
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(features_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # 此处计算参看论文里损失函数公式
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, 
                                                     torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, features, labels=None, predict=None):
        # 克隆一份labels
        labels = labels.unsqueeze(1).float().clone()
        # 插值到特征尺寸
        labels = torch.nn.functional.interpolate(labels, (features.shape[2], features.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        # 维度尺寸完全相同
        assert labels.shape[-1] == features.shape[-1], '{} {}'.format(labels.shape, features.shape)

        batch_size = features.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        features = features.permute(0, 2, 3, 1)
        features = features.contiguous().view(features.shape[0], -1, features.shape[-1])

        features_, labels_ = self._hard_anchor_sampling(features, labels, predict)

        loss = self._contrastive(features_, labels_)
        return loss


class ContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastCELoss, self).__init__()
        self.configer = configer
        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')
        self.seg_criterion = FSCELoss(configer=configer)
        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        assert 'seg' in preds
        assert "embed" in preds
        h, w = target.size(1), target.size(2)               
        seg = preds['seg']
        embedding = preds['embed']
        # 插值/缩放到目标尺寸
        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        # 先计算分类交叉熵损失函数
        loss = self.seg_criterion(pred, target)
        # 按行取preds预测结果里的最大值
        _, predict = torch.max(seg, 1)
        # 计算像素对比损失值
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        
        # return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training
    

class ContrastAuxCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastAuxCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.seg_criterion = FSAuxCELoss(configer=configer)
        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "seg_aux" in preds
        assert "embed" in preds

        seg = preds['seg']
        seg_aux = preds['seg_aux']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion([pred_aux, pred], target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training

