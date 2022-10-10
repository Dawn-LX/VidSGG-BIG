import json
import os
from tqdm import tqdm
from collections import defaultdict
import pickle

def vidor():
    dir_vidor_train = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation/training"
    dir_vidor_val = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation/validation"

    anno_dir = dir_vidor_train
    ann_file_list = os.listdir(anno_dir)




    # "relation_instances": [                         # List of annotated visual relation instances
    #     {
    #         "subject_tid": 0,                       # Corresponding trajectory ID of the subject
    #         "object_tid": 1,                        # Corresponding trajectory ID of the object
    #         "predicate": "move_right", 
    #         "begin_fid": 0,                         # Frame index where this relation begins (inclusive)
    #         "end_fid": 210                          # Frame index where this relation ends (exclusive)
    #     }, 
    #     ...
    # ]

    ## vidor 
    relation_stat = defaultdict(list)
    ann_id_list = os.listdir(anno_dir)
    for idx in tqdm(ann_id_list):
        ann_file_list = os.listdir(os.path.join(anno_dir,idx))
        for ann_name in ann_file_list:
            anno_path = os.path.join(anno_dir,idx,ann_name)
            with open(anno_path, 'r') as fin:
                anno = json.load(fin)
            video_name = anno["video_id"]
            relations = anno["relation_instances"]
            traj_info = anno["subject/objects"]
            tid2category_map = {x["tid"]:x["category"] for x in traj_info}

            for re in relations:
                sub = tid2category_map[re["subject_tid"]]
                obj = tid2category_map[re["object_tid"]]
                pred = re["predicate"]
                triplet = (sub,pred,obj)
                relation_stat[video_name].append(triplet)


    save_name = "statistics/VidORtrain_triplet_bias.pkl"
    with open(save_name,'wb') as f:
        pickle.dump(relation_stat,f)


def vidor_v2():
    dir_vidor_train = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation/training"
    dir_vidor_val = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation/validation"

    anno_dir = dir_vidor_train
    ann_file_list = os.listdir(anno_dir)




    # "relation_instances": [                         # List of annotated visual relation instances
    #     {
    #         "subject_tid": 0,                       # Corresponding trajectory ID of the subject
    #         "object_tid": 1,                        # Corresponding trajectory ID of the object
    #         "predicate": "move_right", 
    #         "begin_fid": 0,                         # Frame index where this relation begins (inclusive)
    #         "end_fid": 210                          # Frame index where this relation ends (exclusive)
    #     }, 
    #     ...
    # ]

    ## vidor 
    relation_stat = defaultdict(list)
    ann_id_list = sorted(os.listdir(anno_dir))
    for gid in tqdm(ann_id_list):
        ann_file_list = os.listdir(os.path.join(anno_dir,gid))
        for ann_name in ann_file_list:
            anno_path = os.path.join(anno_dir,gid,ann_name)
            with open(anno_path, 'r') as fin:
                anno = json.load(fin)
            video_name = gid + "_" + anno["video_id"]
            relations = anno["relation_instances"]
            traj_info = anno["subject/objects"]
            tid2category_map = {x["tid"]:x["category"] for x in traj_info}

            for re in relations:
                sub = tid2category_map[re["subject_tid"]]
                obj = tid2category_map[re["object_tid"]]
                pred = re["predicate"]
                triplet = (sub,pred,obj)
                relation_stat[video_name].append(triplet)


    save_name = "statistics/VidORtrain_triplet_bias.json"
    with open(save_name,'w') as f:
        json.dump(relation_stat,f)


def vidvrd():
    anno_dir = "vidvrd-dataset/train"

    ann_file_list = os.listdir(anno_dir)




    # "relation_instances": [                         # List of annotated visual relation instances
    #     {
    #         "subject_tid": 0,                       # Corresponding trajectory ID of the subject
    #         "object_tid": 1,                        # Corresponding trajectory ID of the object
    #         "predicate": "move_right", 
    #         "begin_fid": 0,                         # Frame index where this relation begins (inclusive)
    #         "end_fid": 210                          # Frame index where this relation ends (exclusive)
    #     }, 
    #     ...
    # ]

    ## vidor 
    relation_stat = defaultdict(list)
    for filename in tqdm(ann_file_list):
        anno_path = os.path.join(anno_dir,filename)
        with open(anno_path, 'r') as fin:
            anno = json.load(fin)
        video_name = anno["video_id"]
        relations = anno["relation_instances"]
        traj_info = anno["subject/objects"]
        tid2category_map = {x["tid"]:x["category"] for x in traj_info}

        for re in relations:
            sub = tid2category_map[re["subject_tid"]]
            obj = tid2category_map[re["object_tid"]]
            pred = re["predicate"]
            triplet = (sub,pred,obj)
            relation_stat[video_name].append(triplet)


    save_name = "statistics/VidVRDtrain_triplet_bias.pkl"
    with open(save_name,'wb') as f:
        pickle.dump(relation_stat,f)


if __name__ == "__main__":
    vidor_v2()