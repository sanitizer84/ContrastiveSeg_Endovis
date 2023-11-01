
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt  
import cv2

from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.distributed import get_world_size, get_rank, is_distributed
from lib.utils.tools.average_meter import AverageMeter
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler


class Trainer(object):
    def __init__(self, configer):
        self.configer = configer
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        # self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        # self.running_score = None
        self._init_model()

    def _init_model(self):
        self.seg_net = self.module_runner.load_net(
            self.model_manager.semantic_segmentor()
        )
        self.val_loader =   self.data_loader.get_valloader()
        self.network_stride = self.configer.get('network', 'stride')


    def __val(self, data_loader=None):
        self.seg_net.eval()
        allrow = None
        count = 0
        if get_rank() != 1: return
        ''' 第一次生成npy文件保存
        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):

            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
            with torch.no_grad():
                outputs = self.seg_net(*inputs, is_eval=True)
                if isinstance(outputs, dict):
                    self.evaluator.update_score(outputs['seg'], data_dict['meta'])
                else:
                    self.evaluator.update_score(outputs, data_dict['meta'])
                    
            outputs = outputs['seg']
            metas = data_dict['meta']

            # ori_img_size = 512

            item = outputs.permute(0, 2, 3, 1)
            item = torch.squeeze(item)
            item = cv2.resize(
                item.cpu().numpy(),
                (512, 512), interpolation=cv2.INTER_CUBIC)
            print('reshaped item:', item.shape)
                    
            item = torch.tensor(item).cuda()
            item = torch.permute(item, [2, 0, 1])
            x = torch.reshape(item, [-1])
            print('x:', x.shape)

            
            t = torch.reshape(targets, [-1])
            t = t.repeat([9])
            print('t:', t.shape)
            
            X = torch.stack((x, t))
            X = torch.permute(X, [1, 0])       

            if allrow is None:
                allrow = X.cpu()
            else:
                allrow = torch.vstack((allrow, X.cpu()))
            count += 1
            if count >50: break

        # tsne  
        allrow1 = allrow[allrow[:, 1] == 1.0][0:2000]
        allrow2 = allrow[allrow[:, 1] == 2.0][0:2000]
        allrow3 = allrow[allrow[:, 1] == 3.0][0:2000]
        allrow4 = allrow[allrow[:, 1] == 4.0][0:2000]
        allrow5 = allrow[allrow[:, 1] == 5.0][0:2000]
        allrow6 = allrow[allrow[:, 1] == 6.0][0:2000]
        allrow7 = allrow[allrow[:, 1] == 7.0][0:2000]
        allrow8 = allrow[allrow[:, 1] == 8.0][0:2000]
        allrow9 = allrow[allrow[:, 1] == 9.0][0:2000]
        print(allrow1.shape)
        print(allrow2.shape)
        print(allrow3.shape)
        print(allrow4.shape)
        print(allrow5.shape)
        print(allrow6.shape)
        print(allrow7.shape)
        print(allrow8.shape)
        print(allrow9.shape)


        allrow = torch.vstack((allrow1, 
                               allrow2, 
                               allrow3, 
                               allrow4, 
                               allrow5, 
                               allrow6, 
                               allrow7,
                               allrow8,
                               allrow9
                               ))
        np.save('allrow.npy', allrow)
        npy文件保存好即可注释
        '''
        
        from tsnecuda import TSNE
        allrow = np.load('/home/duhj/ContrastiveSeg/scripts/davinci/allrow.npy')
      
        # for p in range(700, 1000, 50):
        #     for n in range(140000, 240000, 4000):
        p = 600
        n = 222500
        b = 50
        for t in range(200):
            print(p,' ', n, ' ', b)
            tsne = TSNE(perplexity=p, n_iter=n, num_neighbors=b, learning_rate=500)
            tsne_results = tsne.fit_transform(allrow)
            fig = plt.figure( figsize=(8, 8) )
            ax = fig.add_subplot(1, 1, 1, title='TSNE' )
            scatter  = ax.scatter(
                x=tsne_results[:, 0],
                y=tsne_results[:, 1],
                c=allrow[:, 1],
                cmap=plt.cm.get_cmap('rainbow'),
                s=2
                )
            legend1 = ax.legend(*scatter.legend_elements(), loc="best")       
            ax.add_artist(legend1)
            plt.axis('off')
            plt.savefig('TSNE-p'+str(p)+'-n'+str(n)+'-b'+str(b)+'-'+str(t)+'.png')
            plt.close(fig)


    def train(self):
        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))


if __name__ == "__main__":
    pass
