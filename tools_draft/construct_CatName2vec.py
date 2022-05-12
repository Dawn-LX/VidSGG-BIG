
import root_path
import numpy as np
from tqdm import tqdm

from utils.categories_v2 import vidvrd_CatId2name,vidvrd_PredId2name,vidor_CatId2name,vidor_PredId2name,PKU_vidvrd_CatId2name

file_path = "/home/gkf/project/2D-TAN/.vector_cache/glove.6B.300d.txt"  # 6B means 6 billions
with open(file_path,'r') as f:
    glove6B_300d = f.readlines()
glove6B_300d = [line.strip().split(" ") for line in glove6B_300d]
print(len(glove6B_300d))

glove6B_300d_dict = {}
words_list = []
for word2vec in tqdm(glove6B_300d):
    assert len(word2vec) == 301, "len(word2vec)={}".format(len(word2vec))
    word = word2vec[0]
    words_list.append(word)
    vector = [float(x) for x in word2vec[1:]]
    glove6B_300d_dict[word] = np.array(vector) # shape == (300,)

def get_wordvec(word):
    global glove6B_300d_dict
    words = word.split('_')
    if len(words) == 1:
        vector = glove6B_300d_dict[word]
    elif len(words) ==2:
        w1,w2 = words
        vector = (glove6B_300d_dict[w1] + glove6B_300d_dict[w2])/2
    elif len(words) ==3:
        w1,w2,w3 = words
        vector = (glove6B_300d_dict[w1] + glove6B_300d_dict[w2] + glove6B_300d_dict[w3])/3
    else:
        print(words)
        assert False
        
    return vector

def construct_vidvrd_entity():
    num_enti = len(vidvrd_CatId2name)
    num_pred = len(vidvrd_PredId2name)
    assert num_enti == 36 and num_pred == 133

    ## entity name2vec:
    enti_matrix = np.zeros(shape=(num_enti,300))
    for idx,name in vidvrd_CatId2name.items():
        if name == "__background__":
            vector = np.zeros(shape=(300,))
            enti_matrix[idx] = vector
            continue

        names = name.split('/')
        if len(names) == 2:
            n1,n2 = names
            vector = (get_wordvec(n1) + get_wordvec(n2)) / 2
        elif len(names) == 1:
            vector = get_wordvec(name)
        else:
            assert False
        
        enti_matrix[idx] = vector
    np.save("tools/vidvrd_EntiNameEmb.npy",enti_matrix)

    pred_matrix = np.zeros(shape=(num_pred,300))
    for idx,name in vidvrd_PredId2name.items():
        if name == "__background__":
            vector = np.zeros(shape=(300,))
            pred_matrix[idx] = vector
            continue

        names = name.split('/')
        if len(names) == 2:
            n1,n2 = names
            vector = (get_wordvec(n1) + get_wordvec(n2)) / 2
        elif len(names) == 1:
            vector = get_wordvec(name)
        else:
            assert False
        
        pred_matrix[idx] = vector
    np.save("tools/vidvrd_PredNameEmb.npy",pred_matrix)


def construct_vidvrd_entity_pku():
    num_enti = len(PKU_vidvrd_CatId2name)
    num_pred = len(vidvrd_PredId2name)
    assert num_enti == 36 and num_pred == 133

    ## entity name2vec:
    enti_matrix = np.zeros(shape=(num_enti,300))
    for idx,name in vidvrd_CatId2name.items():
        if name == "__background__":
            vector = np.zeros(shape=(300,))
            enti_matrix[idx] = vector
            continue

        names = name.split('/')
        if len(names) == 2:
            n1,n2 = names
            vector = (get_wordvec(n1) + get_wordvec(n2)) / 2
        elif len(names) == 1:
            vector = get_wordvec(name)
        else:
            assert False
        
        enti_matrix[idx] = vector
    np.save("tools/vidvrd_EntiNameEmb_pku.npy",enti_matrix)

   


def construct_vidor_NameEmb():
    num_enti = len(vidor_CatId2name)
    num_pred = len(vidor_PredId2name)
    assert num_enti == 81 and num_pred == 51
    
    ## entity name2vec:
    enti_matrix = np.zeros(shape=(num_enti,300))
    for idx,name in vidor_CatId2name.items():
        if name == "__background__":
            vector = np.zeros(shape=(300,))
            enti_matrix[idx] = vector
            continue

        names = name.split('/')
        if len(names) == 2:
            n1,n2 = names
            vector = (get_wordvec(n1) + get_wordvec(n2)) / 2
        elif len(names) == 1:
            vector = get_wordvec(name)
        else:
            assert False
        
        enti_matrix[idx] = vector
    np.save("tools/vidor_EntiNameEmb.npy",enti_matrix)

    ## predicate name2vec
    pred_matrix = np.zeros(shape=(num_pred,300))
    for idx,name in vidor_PredId2name.items():
        if name == "__background__":
            vector = np.zeros(shape=(300,))
            pred_matrix[idx] = vector
            continue
        
        if name == "play(instrument)":
            name = "play_instrument"
        
        vector = get_wordvec(name)
        pred_matrix[idx] = vector

    np.save("tools/vidor_PredNameEmb.npy",pred_matrix)

if __name__ == "__main__":
    construct_vidvrd_entity_pku()
    
    
