# ====================================================
# This code is added by Kaifeng Gao (kite_phone@zju.edu.cn)
# ====================================================


import json
from dataset import VidVRD, VidOR


def prepare_gts_for_vidvrd():
    dataset = VidVRD(
        anno_rpath = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset", 
        video_rpath = '/home/gkf/project/VidVRD_VidOR/vidvrd-dataset/videos', 
        splits = ["train","test"],
    )
    indices = dataset.get_index(split="test")


    video_level_gts = dict()
    for vid in indices:
        video_level_gts[vid] = dataset.get_relation_insts(vid)
    save_path = "datasets/GT_json_for_eval/VidVRDtest_gts.json"
    with open(save_path,'w') as f:
        json.dump(video_level_gts,f)


def prepare_gts_for_vidor():
    dataset = VidOR(
        anno_rpath = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation", 
        video_rpath = '/home/gkf/project/VidVRD_VidOR/vidor-dataset/val_videos', # videos for val
        # splits = ["training","validation"],
        splits = ["validation"],
        low_memory=False
    )
    indices = dataset.get_index(split="validation")


    video_level_gts = dict()
    for vid in indices:
        video_level_gts[vid] = dataset.get_relation_insts(vid)
    save_path = "datasets/GT_json_for_eval/VidORval_gts.json"
    with open(save_path,'w') as f:
        json.dump(video_level_gts,f)


if __name__ == "__main__":
    # prepare_gts_for_vidvrd()
    prepare_gts_for_vidor()
    
    