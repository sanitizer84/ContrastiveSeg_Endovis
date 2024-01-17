import numpy as np
import torch
# import torch.nn as nn
import matplotlib.pyplot as plt  

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank   #, is_distributed

from matplotlib.colors import ListedColormap
colors = ["#00ff00", "#33ffff", "#7dff0c", "#ff3700", "#18377d", "#bb9b19", "#38aff8", "#f880ed"]
my_cmap = ListedColormap(colors, name="my_cmap")

'''tsne cuda'''        
from tsnecuda import TSNE

class Trainer(object):
    # The class for Pose Estimation. Include train, val, val & predict.
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        # self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        # self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        
        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()
        self.pixel_loss = self.module_runner.to_device(self.pixel_loss)
    
    def __val(self, data_loader=None):
        self.seg_net.eval()
        self.pixel_loss.eval()
        allrow = None
        
        # Do this once
        data_loader = self.val_loader if data_loader is None else data_loader
        # count = 0
        # for j, data_dict in enumerate(data_loader):
        #     # self.optimizer.zero_grad()
        #     (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
        #     with torch.no_grad():
        #         outputs = self.seg_net(*inputs)
        #         if isinstance(outputs, dict):
        #             try:
        #                 outputs = outputs['pred']
        #             except:
        #                 outputs = outputs['seg']

        #     x = torch.squeeze(outputs)
        #     x = torch.reshape(x, [-1])
        #     t = torch.reshape(targets, [-1])
            
        #     x = x[:t.shape[0]]
        #     X = torch.stack((x, t))
        #     X = torch.permute(X, [1, 0])       

        #     if allrow is None:
        #         allrow = X.cpu()
        #     else:
        #         allrow = torch.vstack((allrow, X.cpu()))
        #     count += 1
        #     if count >9: break


        
        # ''' tsne'''       
        # allrow1 = allrow[allrow[:, 1] == 1.0][0:500]
        # allrow2 = allrow[allrow[:, 1] == 2.0][0:500]
        # allrow3 = allrow[allrow[:, 1] == 3.0][0:500]
        # allrow4 = allrow[allrow[:, 1] == 4.0][0:500]
        # allrow5 = allrow[allrow[:, 1] == 5.0][0:500]
        # allrow6 = allrow[allrow[:, 1] == 6.0][0:500]
        # allrow7 = allrow[allrow[:, 1] == 7.0][0:500]
        # allrow8 = allrow[allrow[:, 1] == 8.0][0:500]
        # print(allrow1.shape)
        # print(allrow2.shape)
        # print(allrow3.shape)
        # print(allrow4.shape)
        # print(allrow5.shape)
        # print(allrow6.shape)
        # print(allrow7.shape)
        # print(allrow8.shape)


        # allrow = torch.vstack((allrow1, 
        #                        allrow2, 
        #                        allrow3, 
        #                        allrow4, 
        #                        allrow5, 
        #                        allrow6, 
        #                        allrow7,
        #                        allrow8
        #                        ))
        # np.save('allrow-old500.npy', allrow)
        # # do this once

        allrow = np.load('allrow-old500.npy') 
        n = 100000
        for p in range(1400, 400, -100):
            for b in range(500, 100, -100): 
                tsne_results =  TSNE(perplexity=p, 
                                    n_iter=n, 
                                    learning_rate=300,
                                    num_neighbors=b,
                                    n_iter_without_progress=4000,
                                    random_seed=5
                                    ).fit_transform(allrow)   
                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(
                    x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    c=allrow[:, 1],
                    cmap=plt.cm.get_cmap(my_cmap),
                    s=2
                    )
                plt.axis('off')
                plt.savefig('/home/duhj/tsne/p'+str(p)+'-n'+str(n)+'-b'+str(b)+'.png')
                plt.close(fig)
                print(p, n, b)

    def train(self):
        print('start tsne...')
        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))


if __name__ == "__main__":
    pass
