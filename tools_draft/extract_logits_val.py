import numpy as np
import os
from numpy.core.fromnumeric import sort
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


    state_dict = torch.load(weight,map_location=torch.device('cpu'))
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
    cls_model = create_model()
    device = torch.device("cuda:0")
    cls_model = cls_model.cuda(device)

    load_dir = "/home/gkf/project/deepSORT/tracking_results/nms0.5_miss30_score0.3/VidORval_freq1"
    save_dir = "/home/gkf/project/deepSORT/tracking_results/nms0.5_miss30_score0.3/VidORval_freq1_logits"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # res_path_list = []  # for tr
    # for part_id in range(1,15):
    #     part_name = "VidORtrain_freq1_part{:02d}".format(part_id)
    #     part_dir = os.path.join(load_dir,part_name)
    #     paths = sorted(os.listdir(part_dir))
    #     paths = [os.path.join(part_dir,p) for p in paths]
    #     res_path_list += paths
    # assert len(res_path_list) == 7000

    res_path_list = sorted(os.listdir(load_dir))
    res_path_list = [os.path.join(load_dir,r) for r in res_path_list]
    assert len(res_path_list) == 835
    for load_path in tqdm(res_path_list):
        # print(load_path)
        track_res = np.load(load_path,allow_pickle=True)
        batch_features = []
        for box_info in track_res:
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 6 or len(box_info) == 12 + dim_feature,"len(box_info)=={}".format(len(box_info))
            
            if len(box_info) == 12 + dim_feature:
                cat_id = box_info[7]
                roi_feature = box_info[12:]
                batch_features.append(roi_feature)
                assert cat_id > 0
            else:
                batch_features.append([0]*dim_feature)
        
        batch_features = torch.tensor(batch_features).float()
        assert len(track_res) == batch_features.shape[0]
    
        cls_logits = cls_model(batch_features.to(device))  # shape == (N,81)
        cls_logits = cls_logits.cpu().numpy()
        save_path = os.path.join(
            save_dir,load_path.split('/')[-1].split('.')[0] + "_logits.npy"
        ) 
        np.save(save_path,cls_logits)
        # print(save_path)
        # break

    print("finish")

