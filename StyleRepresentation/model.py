'''
Code imported from https://github.com/minzwon/data-driven-harmonic-filters
'''

# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules import HarmonicSTFT
from modules import ResNet_StyleExtractor_1 as ResNet

class SingleExtractor(nn.Module):
    def __init__(self, 
                conv_channels=128,
                sample_rate=16000,
                n_fft=513,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None,
                dataset='mtat'):
        super(SingleExtractor, self).__init__()

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)
    
    def forward(self, x):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x))

        # 2D CNN
        representations = self.conv_2d(x)

        return representations
