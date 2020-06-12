import os
import time
import numpy as np

import torch 
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchaudio

from model import SingleExtractor
from loss import TripletLoss


class Trainer(object):
    def __init__(self, dataloader, negative_sample_size=4, \
                 n_epochs=500, loss_mode='cosine', \
                 starting_lr=1e-3, device='cpu'):
        self.model = SingleExtractor(conv_channels=128,
                                     sample_rate=16000,
                                     n_fft=513,
                                     n_harmonic=6,
                                     semitone_scale=2,
                                     learn_bw='only_Q').to(device)
        self.device = device
        self.negative_sample_size = negative_sample_size
        self.n_epochs = n_epochs
        self.criterion = TripletLoss(mode=loss_mode, device=self.device)
        self.optimizer = Adam(self.model.parameters(), lr=starting_lr, weight_decay=1e-4)
        self.current_optimizer = 'adam'
        self.drop_counter = 0
        self.trianing_loss = []
        self.best_train_loss = 100
        self.model_save_path = 'checkpoints'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.dataloader = dataloader
         
    def optimizerScheduler(self):
        # Adam to sgd
        if self.current_optimizer == 'adam' and self.drop_counter == 60:
            self.optimizer = SGD(self.model.parameters(), 1e-3, momentum=0.9, weight_decay=0.0001, nesterov=True)
            self.current_optimizer = 'sgd_1'
            self.drop_counter = 0
            print('sgd 1e-3')
        # First drop
        elif self.current_optimizer == 'sgd_1' and self.drop_counter == 20:
            for pg in self.optimizer.param_groups:
                pg['lr'] = 1e-4
            self.current_optimizer = 'sgd_2'
            self.drop_counter = 0
            print('sgd 1e-4')
        # Second drop
        elif self.current_optimizer == 'sgd_2' and self.drop_counter == 20:
            for pg in self.optimizer.param_groups:
                pg['lr'] = 1e-5
            self.current_optimizer = 'sgd_3'
            print('sgd 1e-5')

            
    def train(self):
        t0 = time.time()
        for epoch in range(self.n_epochs):
            self.drop_counter += 1
            self.model.train()
            epoch_loss = []
            for i, (anchor, pos, negs) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                anchor = Variable(anchor).to(self.device)
                pos = Variable(pos).to(self.device)
                # shape: (1, negative_sample_size, x, x) => (negative_sample_size, x, x)
                # cannot handle when batch_size is not 1
                negs = negs.squeeze(0)
                negs = Variable(negs).to(self.device)
                
                # Feed tensors into the Siamese harmonic network
                ha = self.model(anchor)
                hp = self.model(pos)
                hn = self.model(negs)
                # print (ha.shape, hp.shape, hn.shape)
                # Compute triplet loss
                loss = self.criterion(ha, hp, hn)
                # print (loss.device)
                epoch_loss.append(loss.item())
                
                # print (i, loss.item())
                loss.backward()
                self.optimizer.step()
                
                if epoch_loss[-1] < self.best_train_loss:
                    self.best_train_loss = epoch_loss[-1]
                    torch.save(self.model.state_dict(),\
                               os.path.join(self.model_save_path, f'best_training_model_epoch{epoch}_iter{i}_loss{epoch_loss[-1]}.pth'))
                    
            self.trianing_loss.append(np.mean(epoch_loss))
            self.optimizerScheduler()
            
            print ("Epoch: {:3d} | Train loss: {:.5f} | Time: {:4d}s".format(epoch, self.trianing_loss[-1], int(time.time()-t0)))
