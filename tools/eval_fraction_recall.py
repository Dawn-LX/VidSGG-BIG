import root_path

import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch

from dataloaders.dataloader_vidor_v3 import Dataset
from utils.utils_func import parse_config_py,create_logger
from utils.utils_func import unique_with_idx_nd

def prepare_gt_data(gt_graph):
    if gt_graph.num_trajs==0 or gt_graph.num_preds==0:
        return None
    
    video_len = gt_graph.video_len
    # traj_bboxes = gt_graph.traj_bboxes  # list[tensor],each shape == (n_frames,4) # format: xyxy
    traj_bboxes = gt_graph.traj_bboxes  # list[tensor],each shape == (n_frames,4) # format: xyxy
    traj_cats = gt_graph.traj_cat_ids  # shape == (n_traj,)
    traj_duras = gt_graph.traj_durations  # shape == (n_traj,2)
    pred_durations = gt_graph.pred_durations  # shape == (n_pred,2)
    pred_cats  = gt_graph.pred_cat_ids     # shape == (n_pred,)
    pred2so_ids = torch.argmax(gt_graph.adj_matrix,dim=-1).t()  # enti index,  shape == (n_gt_pred,2)
    pred2so_cats = traj_cats[pred2so_ids] # shape == (n_pred,2)


    gt_5tuples = torch.cat(
        [pred_cats[:,None],pred2so_cats,pred2so_ids],dim=-1
    ) # (n_pred,5)  format: [pred_catid,subj_catid,obj_catid,s_id,o_id]

    return gt_5tuples

def calculate_n_hits(gt_5tuple,gt2hit_ids,n_recall):
    if gt_5tuple.shape[0] == 0:
        return 0,{k:0.0 for k in n_recall}
    _,index_map = unique_with_idx_nd(gt_5tuple)
    n_uniq_gt = len(index_map)
    # n_uniq_gt = gt_5tuple.shape[0]

    n_hits_k = {}
    for k in n_recall:
        n_hit = 0.0
        for im in index_map:
            hit_ids = gt2hit_ids[im]
            n_hit += ((hit_ids >= 0) & (hit_ids <= k)).sum().float().item() / len(im)
        n_hits_k[k] = n_hit
    
    return n_uniq_gt,n_hits_k


def evaluate(cfg_path,hit_info_path,experiment_dir=None):
    def reset_video_name(video_name):
        temp = video_name.split('_')  # e.g., "0001_3598080384"
        assert len(temp) == 2
        video_name = temp[1]
        return video_name
    

    ## create dirs
    if experiment_dir == None:
        experiment_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(experiment_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = hit_info_path.split('/')[-1] + '.log'
    log_path = os.path.join(log_dir,log_filename)
    logger = create_logger(log_path)
    

    all_cfgs = parse_config_py(cfg_path)
    dataset_config = all_cfgs["test_dataset_config"]

    ## construct dataset
    dataset = Dataset(**dataset_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        collate_fn=dataset.collator_func,
        shuffle=False,
        num_workers=2
    )

    with open(hit_info_path,'rb') as f:
        hit_infos = pickle.load(f)
    logger.info("hit_info_path loaded from {}".format(hit_info_path))
    
    n_recalls = [50,100,150,1000]
    total_hit = {n:0 for n in n_recalls}
    total_hit1 = {n:0 for n in n_recalls}
    total_hit2 = {n:0 for n in n_recalls}
    recall_at_n = {n:[] for n in n_recalls}
    recall_at_n1 = {n:[] for n in n_recalls}
    recall_at_n2 = {n:[] for n in n_recalls}

    total_gt = 0
    total_gt1 = 0
    total_gt2 = 0
    for _,_,gt_graph_list in tqdm(dataloader):
        assert len(gt_graph_list) == 1
        gt_graph = gt_graph_list[0]
        gt_5tuple = prepare_gt_data(gt_graph)

        try:
            hit_scores,gt2hit_ids = hit_infos[reset_video_name(gt_graph.video_name)]
        except KeyError:
            continue
            
        hit_scores = torch.from_numpy(hit_scores)
        gt2hit_ids = torch.from_numpy(gt2hit_ids)  # shape == (n_gt,)

        ## for all
        n_gt,n_hits =  calculate_n_hits(gt_5tuple,gt2hit_ids,n_recalls)
        total_gt += n_gt
        for k,v in n_hits.items():
            total_hit[k] += v
            if n_gt >0:
                recall_at_n[k].append(v/n_gt)

        uniq_gt5tuple,index_map = unique_with_idx_nd(gt_5tuple)
        
        ids_single5 = [torch.empty(size=(0,),dtype=torch.long)]
        ids_multiple5 = [torch.empty(size=(0,),dtype=torch.long)]
        for im in index_map:
            if len(im) == 1:
                ids_single5.append(im)
            else:
                ids_multiple5.append(im)
        
        ids_single5 = torch.cat(ids_single5,dim=0)
        ids_multiple5 = torch.cat(ids_multiple5,dim=0)

        uniq_singles = gt_5tuple[ids_single5,:]
        uniq_multiples = gt_5tuple[ids_multiple5,:]

        gt2hit_ids_for_single = gt2hit_ids[ids_single5]
        gt2hit_ids_for_multiple = gt2hit_ids[ids_multiple5]


        ## for single
        n_gt,n_hits =  calculate_n_hits(uniq_singles,gt2hit_ids_for_single,n_recalls)
        total_gt1 += n_gt
        for k,v in n_hits.items():
            total_hit1[k] += v
            if n_gt >0:
                recall_at_n1[k].append(v/n_gt)

        
        ## for multiple
        n_gt,n_hits =  calculate_n_hits(uniq_multiples,gt2hit_ids_for_multiple,n_recalls)
        total_gt2 += n_gt
        for k,v in n_hits.items():
            total_hit2[k] += v
            if n_gt >0:
                recall_at_n2[k].append(v/n_gt)

    logger.info("---------------video-level----------------")
    recall_at_n = {k:np.mean(v) for k,v in recall_at_n.items()}
    recall_at_n1 = {k:np.mean(v) for k,v in recall_at_n1.items()}
    recall_at_n2 = {k:np.mean(v) for k,v in recall_at_n2.items()}
    logger.info(f"overall{recall_at_n}")
    logger.info(f"single{recall_at_n1}")
    logger.info(f"multiple{recall_at_n2}")


    logger.info("---------------dataset-level----------------")
    recall_at_n = {k:v/total_gt for k,v in total_hit.items()}
    recall_at_n1 = {k:v/total_gt1 for k,v in total_hit1.items()}
    recall_at_n2 = {k:v/total_gt2 for k,v in total_hit2.items()}
    logger.info(f"overall{recall_at_n}")
    logger.info(f"single{recall_at_n1}")
    logger.info(f"multiple{recall_at_n2}")

    logger.info("log file has been saved at {}".format(log_path))
    logger.handlers.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--hit_info_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    
    args = parser.parse_args()

    evaluate(
        args.cfg_path,
        hit_info_path = args.hit_info_path,
        experiment_dir=args.output_dir
    )
    '''
    ### table-6 #Bins=1
    
    python tools/eval_fraction_recall.py \
        --cfg_path experiments/grounding_weights/config_bin1.py \
        --hit_info_path  experiments/exp5_with_grounding/VidORval_hit_infos_aft_grd_with_grd_epoch70_bin1.pkl
    
    2022-03-14 05:48:24,512 - ---------------video-level----------------
    2022-03-14 05:48:24,515 - overall{50: 0.181108409023905, 100: 0.20063033702851685, 150: 0.20931486009123837, 1000: 0.21292891505127884}
    2022-03-14 05:48:24,515 - single{50: 0.19312158069101806, 100: 0.21494420557838229, 150: 0.2255823820086578, 1000: 0.23036639483464477}
    2022-03-14 05:48:24,516 - multiple{50: 0.06842535859009717, 100: 0.08144234313398983, 150: 0.08744749767682543, 1000: 0.09074592122506646}
    2022-03-14 05:48:24,516 - ---------------dataset-level----------------
    2022-03-14 05:48:24,516 - overall{50: 0.10472581583249496, 100: 0.12666797224318133, 150: 0.13646721911329088, 1000: 0.1418562590822714}
    2022-03-14 05:48:24,516 - single{50: 0.1296464939150592, 100: 0.15597317658746584, 150: 0.16764632833843862, 1000: 0.17360708667936087}
    2022-03-14 05:48:24,516 - multiple{50: 0.055354506900349106, 100: 0.06861031056127001, 150: 0.0746970927674552, 1000: 0.0789534795931384}
    2022-03-14 05:48:24,516 - log file has been saved at experiments/grounding_weights/logfile/VidORval_hit_infos_aft_grd_with_grd_epoch70_bin1.pkl.log
    

    ### table-6 #Bins=5
    
    python tools/eval_fraction_recall.py \
        --cfg_path experiments/grounding_weights/config_bin5.py \
        --hit_info_path  experiments/exp5_with_grounding/VidORval_hit_infos_aft_grd_with_grd_epoch70_bin5.pkl
    
    2022-03-14 05:48:50,104 - ---------------video-level----------------
    2022-03-14 05:48:50,107 - overall{50: 0.18533941835849108, 100: 0.20797144472520077, 150: 0.21936553741422612, 1000: 0.2325351539240147}
    2022-03-14 05:48:50,107 - single{50: 0.19909501650719616, 100: 0.22357906720914097, 150: 0.23478592465617895, 1000: 0.24725686214808415}
    2022-03-14 05:48:50,107 - multiple{50: 0.07585911999953157, 100: 0.09248164558407562, 150: 0.10276992698965223, 1000: 0.12089873123862116}
    2022-03-14 05:48:50,107 - ---------------dataset-level----------------
    2022-03-14 05:48:50,107 - overall{50: 0.10616452621981887, 100: 0.12940507629107986, 150: 0.14178862235915493, 1000: 0.15779592922116023}
    2022-03-14 05:48:50,107 - single{50: 0.13072274194883682, 100: 0.15837403758589289, 150: 0.17269641526616442, 1000: 0.18892292408311948}
    2022-03-14 05:48:50,107 - multiple{50: 0.05751130532580426, 100: 0.07201355858072271, 150: 0.08055601115302603, 1000: 0.0961290486343791}
    2022-03-14 05:48:50,108 - log file has been saved at experiments/grounding_weights/logfile/VidORval_hit_infos_aft_grd_with_grd_epoch70_bin5.pkl.log
    
    ### table-6 #Bins=10
    
    python tools/eval_fraction_recall.py \
        --cfg_path experiments/grounding_weights/config_.py \
        --hit_info_path  experiments/exp5_with_grounding/VidORval_hit_infos_aft_grd_with_grd_epoch70.pkl

    2022-03-14 00:00:23,437 - ---------------video-level----------------
    2022-03-14 00:00:23,440 - overall{50: 0.18567408163460938, 100: 0.21118331176903204, 150: 0.22527608059540427, 1000: 0.25118945648191804}
    2022-03-14 00:00:23,440 - single{50: 0.19962606591707377, 100: 0.22728338862158914, 150: 0.24217566960749706, 1000: 0.26565030837554243}
    2022-03-14 00:00:23,440 - multiple{50: 0.0766374992843449, 100: 0.09787206030896713, 150: 0.11142509115970176, 1000: 0.15133007605839785}
    2022-03-14 00:00:23,440 - ---------------dataset-level----------------
    2022-03-14 00:00:23,440 - overall{50: 0.10602492996171468, 100: 0.1301212221244132, 150: 0.1447708405578674, 1000: 0.17373204465111383}
    2022-03-14 00:00:23,440 - single{50: 0.13047437701796508, 100: 0.15895355575792697, 150: 0.17617352429836908, 1000: 0.20341087838397218}
    2022-03-14 00:00:23,440 - multiple{50: 0.05758719484732277, 100: 0.07300038270187523, 150: 0.08255778218464782, 1000: 0.11493417149067464}
    2022-03-14 00:00:23,440 - log file has been saved at experiments/grounding_weights/logfile/VidORval_hit_infos_aft_grd_with_grd_epoch70.pkl.log
    '''
