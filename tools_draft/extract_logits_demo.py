import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
torch.set_printoptions(sci_mode=False,precision=4)
class ClsFC(nn.Module):
    def __init__(self,num_cls,in_dim):
        super(ClsFC,self).__init__()
        self.fc = nn.Linear(in_dim,num_cls)

    @torch.no_grad()
    def forward(self,x):
        return self.fc(x)

def create_model():
    weight = "training_dir/COCO34ORfreq32_4gpu/model_0180000.pth"
    # the weight has been released,
    # refer to https://github.com/Dawn-LX/VidVRD-tracklets#quick-start


    state_dict = torch.load(weight)
    state_dict = state_dict["model"]
    # print(state_dict.keys())

    cls_state_dict = {
        "fc.weight":state_dict['module.roi_heads.box.predictor.cls_score.weight'].cpu(),
        "fc.bias":state_dict['module.roi_heads.box.predictor.cls_score.bias'].cpu()
    }




    model = ClsFC(81,1024)
    model.load_state_dict(cls_state_dict)

    return model

if __name__ == "__main__":
    dim_feature = 1024
    num_cls = 81
    #NOTE originally in 10.12.86.103
    loadfile = "/home/gkf/deepSORT/tracking_results/miss60_minscore0p3/VidORtrain_freq1_part01/0000_2401075277.npy"
    track_res = np.load(loadfile,allow_pickle=True)
    batch_features = []
    cat_ids = []
    scores = []
    for box_info in track_res:
        if not isinstance(box_info,list):
            box_info = box_info.tolist()
        assert len(box_info) == 6 or len(box_info) == 12 + dim_feature,"len(box_info)=={}".format(len(box_info))
        
        if len(box_info) == 12 + dim_feature:
            score = box_info[6]
            cat_id = box_info[7]
            roi_feature = box_info[12:]
            batch_features.append(roi_feature)
            cat_ids.append(cat_id)
            scores.append(score)
            assert cat_id > 0
        else:
            batch_features.append([0]*dim_feature)
            cat_ids.append(0)
            scores.append(0)
    
    cat_ids = torch.tensor(cat_ids)
    scores = torch.tensor(scores)
    batch_features = torch.tensor(batch_features).float()
    assert len(track_res) == len(cat_ids)
    print(cat_ids.shape,batch_features.shape)

    cls_model = create_model()
    cls_logits = cls_model(batch_features)
    cls_logits = torch.softmax(cls_logits,dim=-1)
    cls_logits[:,0] = -1
    print(cls_logits.shape)
    predicted_ids = torch.argmax(cls_logits,dim=-1)
    print(predicted_ids.shape)
    pre_scores = cls_logits[range(7310),predicted_ids]

    assert predicted_ids.shape == cat_ids.shape
    mask = cat_ids > 0
    is_equal = predicted_ids[mask] == cat_ids[mask]
    ratio = is_equal.sum() / len(is_equal)
    print(is_equal,is_equal.shape,ratio)
    

    not_equal = (~ is_equal).nonzero().reshape(-1).tolist()
    index = not_equal[0]
    print(cls_logits[mask][index,:])
    print("fc_cls argmax",predicted_ids[mask][index],pre_scores[mask][index])
    print("detection cls",cat_ids[mask][index],scores[mask][index])


