# ====================================================
# This code is added by Kaifeng Gao (kite_phone@zju.edu.cn)
# ====================================================

import argparse
import json
from dataset import VidVRD, VidOR


def prepare_gts_for_vidvrd(save_path):
    dataset = VidVRD(
        anno_rpath = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset", 
        video_rpath = '/home/gkf/project/VidVRD_VidOR/vidvrd-dataset/videos', 
        splits = ["train","test"],
    )
    indices = dataset.get_index(split="test")


    video_level_gts = dict()
    for vid in indices:
        video_level_gts[vid] = dataset.get_relation_insts(vid)
    
    if save_path is not None:
        print("saving ...")
        with open(save_path,'w') as f:
            json.dump(video_level_gts,f)
        print("done.")


def prepare_gts_for_vidor(save_path):
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
    
    if save_path is not None:
        print("saving ...")
        with open(save_path,'w') as f:
            json.dump(video_level_gts,f)
        print("done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--save_path", type=str,help="...")
    parser.add_argument("--dataset_type", type=str,help="...")
    
    args = parser.parse_args()

    dataset_type = args.dataset_type.lower()
    assert dataset_type in ["vidvrd","vidor"]
    assert args.save_path is None or args.save_path.endswith(".json")

    if dataset_type == "vidvrd":
        prepare_gts_for_vidvrd(args.save_path)
    else:
        prepare_gts_for_vidor(args.save_path)

    '''
    python VidVRD-helper/prepare_gts_for_eval.py \
        --dataset_type vidvrd \
        --save_path datasets/GT_json_for_eval/VidVRDtest_gts.json
    
    python VidVRD-helper/prepare_gts_for_eval.py \
        --dataset_type vidor \
        --save_path datasets/GT_json_for_eval/VidORval_gts.json
    '''


    