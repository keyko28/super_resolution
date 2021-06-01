
import os
import numpy as np
import math
import itertools
import sys
from numpy.core.fromnumeric import std
import psutil

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader, dataloader
from torch.autograd import Variable

from model import SRResNet
from datasets import ImageDataset
from vgg_loss import VGGLoss

from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch
import config as cfg
from tqdm import tqdm


class Train:

    def __init__(self) -> None:

        # load SRresNet and VGG based loss
        self.srresnet = SRResNet()
        self.features = VGGLoss()

        # Set vgg19 to inference mode
        self.features.eval()

        #define loss
        self.loss = torch.nn.L1Loss()

        cuda = torch.cuda.is_available()

        if cuda:

            self.srresnet.cuda()
            self.features.cuda()
            self.loss.cuda()

        #from config
        self.learning_rate = cfg.LERANING_RATE # 0.0002
        self.b1 = cfg.B1 # 0.5
        self.b2 = cfg.B2 # 0.999
        self.shapes = cfg.SHAPES # [256, 256]
        self.mean = cfg.MEAN # np.array([0.485, 0.456, 0.406])
        self.std = cfg.STD # np.array([0.229, 0.224, 0.225])
        self.factor = cfg.DOWNSAMPLE_FACTOR # 2
        self.n_epoch = cfg.EPOCHS # 200
        self.batch_size = cfg.BATCH_SIZE  # 4
        self.sample_interval = cfg.SAMPLE_INTERVAL # 100
        self.batch_save_path = cfg.BATCH_SAVE_PATH 
        self.model_save_path = cfg.MODEL_SAVE_PATH
        self.checkpoint_interval = cfg.CHECKPOINT # 10
        self.data_set_path = cfg.DS_PATH
        self.gamma = cfg.GAMMA # 0.05
        self.max_lr = cfg.MAX_LR # 0.005


        # define ptimizer
        self.optimizer = torch.optim.Adam(self.srresnet.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        
        
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        self.step_size = None
        self.scheduler = None
        self.dataloader = None

        self.workers = psutil.cpu_count(logical=False)

        
    def load_data(self, path: str) -> DataLoader:

        self.dataloader = DataLoader(
            ImageDataset(
                path=path,
                hr_shape=self.shapes,
                mean=self.mean,
                std=self.std,
                factor=self.factor),
                batch_size = self.batch_size,
                shuffle=True,
                num_workers = self.workers 
            )

    def train(self):

        self.load_data(path=self.data_set_path)
        self.step_size = len(self.dataloader) * cfg.STEP_SIZE
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        # self.scheduler = lr_scheduler.CyclicLR(self.optimizer, 
        #                                         base_lr=self.learning_rate, 
        #                                         max_lr=self.max_lr, 
        #                                         step_size_up = self.step_size,
        #                                         mode='triangular2',
        #                                         cycle_momentum=False)

        for epoch in tqdm(range(self.n_epoch)):
            for i, imgs in enumerate(self.dataloader):

                 # setup model input
                imgs_lr = Variable(imgs["lr"].type(self.Tensor))
                imgs_hr = Variable(imgs["hr"].type(self.Tensor))

                self.optimizer.zero_grad()

                gen_hr = self.srresnet(imgs_lr)

                #calc loss over features
                gen_features = self.features(gen_hr)
                real_features = self.features(imgs_hr)
                pixel_loss = self.loss(gen_hr, imgs_hr.detach())
                feature_loss = self.loss(gen_features, real_features.detach())

                loss = pixel_loss + feature_loss

                #backward
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # log
                sys.stdout.write(
                    f"[Epoch: {epoch}/{self.n_epoch}] [Batch {i}/{len(self.dataloader)}] [loss: {loss.item()}] [lr: {self.optimizer.param_groups[0]['lr']}]\n")
                

                batches_complited = epoch * len(self.dataloader) + i
                if batches_complited % self.sample_interval == 0:

                    self._save_image(imgs_lr, imgs_hr, gen_hr, batches_complited)

                if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:

                    # Save model checkpoints
                    self._save_model(epoch=epoch)
                    


    def _save_image(self, lr, hr, gen, batches: int) -> None:

        imgs_lr = nn.functional.interpolate(lr, scale_factor=self.factor)

        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(hr, nrow=1, normalize=True)
        gen_hr = make_grid(gen, nrow=1, normalize=True)

        img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr), -1)

        path = os.path.join(self.batch_save_path, f'batch_{batches}.png')

        save_image(img_grid, path, normalize=False)


    def _save_model(self, epoch: int):

        path = os.path.join(self.model_save_path, f'srsnet_{epoch}.pth' )

        torch.save(self.srresnet.state_dict(), path)

        
        
        

        

