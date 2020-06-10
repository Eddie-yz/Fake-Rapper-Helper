from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class MTATDataset(Dataset):
    def __init__(self, audio_conf, pos_dir, neg_dir=None, negative_sample_size=4):
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.pos_files = os.listdir(pos_dir)
        _sample = np.load(os.path.join(self.pos_dir,self.pos_files[0]))
        self.sample_size = _sample.shape
        self.negative_sample_size = negative_sample_size
        if self.neg_dir is None:
            self.neg_dir = pos_dir
        
        self.neg_files = os.listdir(neg_dir)
        self.size = min(len(self.pos_files) // 2, len(self.neg_files) // negative_sample_size)

    def __getitem__(self, index):
        anchor = np.load(os.path.join(self.pos_dir, self.pos_files[2*index]))
        pos = np.load(os.path.join(self.pos_dir, self.pos_files[2*index+1]))
        negs = []
        for k in range(self.negative_sample_size):
            negs.append(np.load(os.path.join(self.neg_dir, self.neg_files[self.negative_sample_size*index+k])))
        negs = np.array(negs)

        return anchor, pos, negs
    
    def __len__(self):
        return self.size
    
    def shuffle(self):
        np.random.shuffle(self.pos_files)
        if self.neg_dir:
            np.random.shuffle(self.neg_files)