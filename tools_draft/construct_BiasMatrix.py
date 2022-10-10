import pickle
import numpy as np
from collections import defaultdict
# from categories_v2 import vidor_CatName2Id,vidor_PredName2Id
from categories_v2 import vidvrd_CatName2Id,vidvrd_PredName2Id,PKU_vidvrd_CatId2name,PKU_vidvrd_CatName2Id
np.set_printoptions(suppress=True,precision=4,linewidth=1600)

def load_data():
    filename = "/home/gkf/project/VidVRD_VidOR/statistics/VidVRDtrain_triplet_bias.pkl"
    with open(filename,'rb') as f:
        video_triplets = pickle.load(f)

    triplet_list = []
    for video_name, t in video_triplets.items():
        triplet_list += t

    print(len(triplet_list))
    subj_names = list(set([x[0] for x in triplet_list]))
    pred_names = list(set([x[1] for x in triplet_list]))
    obj_names = list(set([x[2] for x in triplet_list]))
    enti_pairs = list(set([(x[0],x[2]) for x in triplet_list]))
    print(len(subj_names),len(pred_names),len(obj_names))

    enti_pairs = defaultdict(list)
    for t in triplet_list:
        enti_pairs[(t[0],t[2])].append(t[1])

    for pair in enti_pairs.keys():
        pred_list = enti_pairs[pair]
        pred_count = defaultdict(int)
        for p in pred_list:
            pred_count[p] += 1
        pred_count["__sum__"] = len(pred_list)
        pred_count = sorted(pred_count.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
        enti_pairs[pair] = pred_count

    print(enti_pairs)
    enti_pairs = sorted(enti_pairs.items(),key=lambda kv:(kv[1][0][1],kv[0]),reverse=True)

    return enti_pairs

CatName2Id = PKU_vidvrd_CatName2Id  ############# NOTE PKU
PredName2Id = vidvrd_PredName2Id

n_enti = len(CatName2Id.keys())   # include background
n_pred = len(PredName2Id.keys())  # include background

bias_matrix = np.zeros(shape=(n_enti,n_enti,n_pred))

enti_pairs = load_data()
for pair,pred_list in enti_pairs:
    s_id,o_id = CatName2Id[pair[0]],CatName2Id[pair[1]]
    for p in pred_list:
        if p[0] == "__sum__":
            continue
        p_id = PredName2Id[p[0]]
        bias_matrix[s_id,o_id,p_id] = p[1]
bias_matrix[:,:,0] += 1
pred_sum = bias_matrix.sum(axis=-1)
bias_matrix = bias_matrix / pred_sum[:,:,np.newaxis]
bias_matrix = np.log(bias_matrix + 1e-3)
print(bias_matrix.shape,pred_sum.shape) 
print(bias_matrix[5,5,:10])
print(bias_matrix[5,8,:10])
print(bias_matrix[0,0,:10])
np.save("statistics/pred_bias_matrix_vidvrd_pku.npy",bias_matrix)


