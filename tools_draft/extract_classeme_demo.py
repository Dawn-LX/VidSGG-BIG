import root_path
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
np.set_printoptions(suppress=True,precision=4,linewidth=500)

from utils.categories_v2 import vidor_categories

loadpath = "tools/vidor_CatName2vec_dict.pkl"  # a dict version of `vidor_EntiNameEmb.npy`, refer to `tools/construct_CatName2vec.py`
with open(loadpath,'rb') as f:
    vidor_CatName2Vec = pickle.load(f)


vidor_CatNames = [v["name"] for v in vidor_categories]

loadpath = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1_logits/0001_2793806282_logits.npy"
logits = np.load(loadpath)
print(logits.shape)

word_emb = [vidor_CatName2Vec[name] for name in vidor_CatNames]
word_emb = np.stack(word_emb,axis=0)
print(word_emb.shape)

def demo():
    # 对每个类别的 embedding 加权平均，就是做个矩阵乘法
    logits = np.random.rand(3,5)
    embs = np.random.randint(1,9,size=(5,7))

    print(logits,logits.shape)
    print(embs,embs.shape)
    res = np.dot(logits,embs)
    print(res,res.shape)

    res2 = []
    for lo in logits:
        assert lo.shape == (5,)
        xx = embs * lo[:,np.newaxis]
        xx = np.sum(xx,axis=0)  # shape == (7,)
        res2.append(xx)
    res2 = np.stack(res2,axis=0)
    print(res2,res2.shape)

logits = logits[:,1:]       
logits = torch.from_numpy(logits)
probs = torch.softmax(logits,dim=-1).numpy() # shape == (N,80)
word_emb = word_emb[1:,:]   # shape == (80,300)
print(probs.shape,word_emb.shape)
print(probs[1,:])
print(logits[1,:].numpy())

weighted_emb = np.dot(probs,word_emb)   # (N,80) x (80,300)
print(weighted_emb.shape)
