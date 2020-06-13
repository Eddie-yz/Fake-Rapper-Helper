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

from dataPrep import MTATDataset
from model import SingleExtractor
from sklearn.manifold import TSNE
import argparse
# In[2]:


device = torch.device(0)


# In[3]:


styEncoder = SingleExtractor(conv_channels=128,
                             sample_rate=16000,
                             n_fft=513,
                             n_harmonic=6,
                             semitone_scale=2,
                             learn_bw='only_Q').to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-loss', type=str, required=True, help='loss type')
parser.add_argument('-margin', type=float, required=True, help='loss type')
parser.add_argument('-cpt', type=str, required=True, help='loss type')
loss = parser.parse_args().loss
marg = parser.parse_args().margin
checkpt = parser.parse_args().cpt

styEncoder.load_state_dict(torch.load(checkpt))


pos_dir = '../Data/spectrogram_pos/'
neg_dir = '../Data/spectrogram_neg/'

audio_embeddings = []
audio_labels = []

os.system("nvidia-smi")

for file in os.listdir(pos_dir):
    file_path = os.path.join(pos_dir, file)
    spec = np.load(file_path)
    emb = styEncoder(Variable(torch.tensor(spec).unsqueeze(0)).to(device)).cpu().detach().numpy().flatten()
    # print (emb.shape)
    audio_embeddings.append(emb)
    audio_labels.append(1)

os.system("nvidia-smi")

for file in os.listdir(neg_dir):
    file_path = os.path.join(neg_dir, file)
    spec = np.load(file_path)
    emb = styEncoder(Variable(torch.tensor(spec).unsqueeze(0)).to(device)).cpu().detach().numpy().flatten()
    # print (emb.shape)
    audio_embeddings.append(emb)
    audio_labels.append(2)
# In[ ]:


audio_embeddings = np.array(audio_embeddings)
audio_labels = np.array(audio_labels)


# In[ ]:


# calculate distance matrix
def calculateCosineDistMat(data):
    num = len(data)
    dist = np.zeros((num, num))
    for i in range(num):
        for j in range(i+1):
            if i == j:
                dist[i, j] = .0
                continue
            cosSim = np.dot(audio_embeddings[i], audio_embeddings[j])/(np.linalg.norm(audio_embeddings[i]) * np.linalg.norm(audio_embeddings[j]))
            d = 10/(1+np.exp(10*cosSim))
            dist[i, j] = d
            dist[j, i] = d
    return dist

def calculateEuclideanDistMat(data):
    num = len(data)
    dist = np.zeros((num, num))
    for i in range(num):
        for j in range(i+1):
            if i == j:
                dist[i, j] = .0
                continue
            d = np.linalg.norm(data[i] - data[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


if loss == "Cosine":
    dist_mat = calculateCosineDistMat(audio_embeddings)
else:
    dist_mat = calculateEuclideanDistMat(audio_embeddings)


# In[ ]:


tsne = TSNE(n_components=2, metric='precomputed', perplexity=50)
X_emb = tsne.fit_transform(dist_mat)


# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(X_emb[:, 0], X_emb[:, 1], c=audio_labels)
cb = plt.colorbar()
cb.set_ticks(np.arange(1,3))
cb.set_ticklabels(['Hiphop', 'Non-Hiphop'])

plt.suptitle('%s loss, margin %.1f' % (loss, marg))

plt.savefig('embedding_%s_%.1f.png' % (loss, marg))
# In[ ]:




