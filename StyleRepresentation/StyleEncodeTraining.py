import os
import time

import torch 
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchaudio

import matplotlib.pyplot as plt
import numpy as np

import argparse

from dataPrep import MTATDataset
from trainer import Trainer

os.system("nvidia-smi")
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('-loss', type=str, required=True, help='loss type')
parser.add_argument('-margin', type=float, required=True, help='loss type')
loss = parser.parse_args().loss
marg = parser.parse_args().margin

train_dataset = MTATDataset(pos_dir='../Data/spectrogram_pos', neg_dir='../Data/spectrogram_neg', negative_sample_size=4)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

styEncTrain = Trainer(dataloader=train_dataloader, device=torch.device("cuda"), loss_mode=loss, starting_lr=1e-4, margin=marg, n_epochs=100)
styEncTrain.train()
