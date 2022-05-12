import root_path
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
np.set_printoptions(precision=4,linewidth=500)

from utils.categories_v2 import vidor_categories

loadpath = "tools/vidor_CatName2vec_dict.pkl" # a dict version of `vidor_EntiNameEmb.npy`, refer to `tools/construct_CatName2vec.py`
with open(loadpath,'rb') as f:
    vidor_CatName2Vec = pickle.load(f)

vidor_CatNames = [v["name"] for v in vidor_categories]
word_emb = [vidor_CatName2Vec[name] for name in vidor_CatNames]
word_emb = np.stack(word_emb,axis=0)
word_emb = word_emb[1:,:] # shape == (80,300)

print(word_emb.shape)

# load_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1_logits/"
# save_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1_classeme/"

load_dir = "/home/gkf/project/MEGA_Pytorch/mega_boxfeatures/GT_boxfeatures/VidORtrain_freq1_logits/"
save_dir = "/home/gkf/project/MEGA_Pytorch/mega_boxfeatures/GT_boxfeatures/VidORtrain_freq1_classeme/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filename_list = sorted(os.listdir(load_dir))
for filename in tqdm(filename_list):
    loadpath = os.path.join(load_dir,filename)
    logits = np.load(loadpath)
    logits = logits[:,1:] # shape == (N,80)
    logits = torch.from_numpy(logits)
    probs = torch.softmax(logits,dim=-1).numpy() # shape == (N,80)
    classeme = np.dot(probs,word_emb) # shape == (N,300)
    
    save_name = filename.split('.')[0].split('logits')[0] + "clsme.npy"
    save_path = os.path.join(save_dir,save_name)
    np.save(save_path,classeme)



