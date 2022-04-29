import root_path
import argparse
import os

import torch
from tqdm import tqdm
from dataloaders.dataloader_vidor import Dataset as Dataset_vidor
from dataloaders.dataloader_vidvrd import Dataset as Dataset_vidvrd
from dataloaders.dataloader_vidvrd import Dataset_pku,Dataset_pku_i3d

from VidVRDhelperEvalAPIs import eval_video_object
from utils.utils_func import create_logger,parse_config_py
from utils.categories_v2 import PKU_vidvrd_CatId2name,vidvrd_CatId2name,vidor_CatId2name



def main(cfg_path,Dataset_class,id2name_map,gt_id2name_map,split="test",part_id=-1):
    
    log_dir = os.path.join(os.path.dirname(cfg_path),'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir,'eval_traj_mAP.log')
    logger = create_logger(log_path)
    
    all_cfgs = parse_config_py(cfg_path)
    dataset_config = all_cfgs["{}_dataset_config".format(split)]

    if Dataset_class == Dataset_vidor and split=="train" and part_id >= 0:
        # for VidOR-train, if part_id is specified, we only use this part.
        train_proposal_dir = dataset_config["proposal_dir"]
        dataset_config["proposal_dir"] = {part_id:train_proposal_dir[part_id]}

    logger.info(dataset_config)
    dataset = Dataset_class(**dataset_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = dataset.collator_func,
        drop_last=False,
        shuffle=False
    )

    proposal_results = {}
    gt_results = {}
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal = proposal_list[0]
        gt_graph = gt_graph_list[0]
        video_name = gt_graph.video_name

        ## convert gt_trajs
        gt_trajbboxes,gt_trajduras = gt_graph.traj_bboxes,gt_graph.traj_durations
        gt_trajduras = gt_trajduras.tolist()
        cat_ids = gt_graph.traj_cat_ids.tolist()
        result_per_video = []
        for cat_id,dura,bboxes in zip(cat_ids,gt_trajduras,gt_trajbboxes):
            frame_ids = list(range(dura[0],dura[1]+1))
            bboxes = bboxes.tolist()
            traj_dict = {fid:box for fid,box in zip(frame_ids,bboxes)}
            result_per_video.append(
                {
                    "category":gt_id2name_map[cat_id],
                    "trajectory":traj_dict
                }
            )
        gt_results.update({video_name:result_per_video})

        ## convert proposal trajs
        # continue
        if proposal.num_proposals == 0:
            proposal_results.update({video_name:[]})
            continue
        pr_trajbboxes,pr_trajduras = proposal.bboxes_list,proposal.traj_durations
        scores,cat_ids = proposal.scores.tolist(),proposal.cat_ids.tolist()
        pr_trajduras = pr_trajduras.tolist()
        result_per_video = []
        for score,cat_id,dura,bboxes in zip(scores,cat_ids,pr_trajduras,pr_trajbboxes):
            frame_ids = list(range(dura[0],dura[1]+1))
            bboxes = bboxes.tolist()
            traj_dict = {fid:box for fid,box in zip(frame_ids,bboxes)}
            result_per_video.append(
                {
                    "category":id2name_map[cat_id],
                    "score":score,
                    "trajectory":traj_dict
                }
            )
        proposal_results.update({video_name:result_per_video})

    mean_ap, ap_class = eval_video_object(gt_results,proposal_results)
    logger.info('=' * 30)
    for i, (category, ap) in enumerate(ap_class):
        logger.info('{:>2}{:>20}\t{:.4f}'.format(i+1, category, ap))
    logger.info('=' * 30)
    logger.info('{:>22}\t{:.4f}'.format('mean AP', mean_ap))

    logger.info(f"log saved at {log_path}")
    logger.handlers.clear()

if __name__ == "__main__":
    # vidor_CatId2name = {idx:"FG" for idx,_ in vidor_CatId2name.items()}
    parser = argparse.ArgumentParser(description='xxxx')
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument('--part_id', type=int,default=-1, help='-1 for all, range: 0~13 for specify one part')
    parser.add_argument('--dataset_type', type=str,default="vidvrd" ,help='...')
    parser.add_argument('--split', type=str,default="test" ,help='...')
    parser.add_argument("--use_pku",action="store_true",default=False,help="...")
    args = parser.parse_args()

    if args.dataset_type.lower() == "vidvrd":
        gt_id2name_map = vidvrd_CatId2name
        if args.use_pku:
            Dataset_class = Dataset_pku
            id2name_map = PKU_vidvrd_CatId2name
        else:
            Dataset_class = Dataset_vidvrd
            id2name_map = vidvrd_CatId2name
    elif args.dataset_type == "vidor":
        gt_id2name_map = vidor_CatId2name
        id2name_map = vidor_CatId2name
        Dataset_class = Dataset_vidor
    else:
        assert False
    
    main(
        cfg_path=args.cfg_path,
        Dataset_class=Dataset_class,
        id2name_map=id2name_map,
        gt_id2name_map = gt_id2name_map,
        split=args.split,
        part_id=args.part_id
    )
    '''
    python tools/eval_traj_mAP.py \
        --cfg_path experiments/exp1/config_.py \
        --dataset_type vidvrd \
        --use_pku
    
    python tools/eval_traj_mAP.py \
        --cfg_path experiments/exp3/config_.py \
        --dataset_type vidvrd
    
    python tools/eval_traj_mAP.py \
        --cfg_path experiments/exp4/config_.py \
        --dataset_type vidor \

    python tools/eval_traj_mAP.py \
        --cfg_path experiments/exp4/config_.py \
        --dataset_type vidor \
        --split train \
        --part_id 0    
    
    datasets/cache/MEGAv9_m60s0.3_freq1_VidORval_freq1_th_15-180-200-0.40.pkl
    '''



# v6_miss60_minscore0p3_VidORval_freq2to1_th_2.pkl
# mean AP  0.0729

# v6_3_miss60_minscore0p3VidORtrain_freq2to1_0_999_th_5-180-200.pkl mean AP  0.1171
# v6_3_miss60_minscore0p3VidORtrain_freq2to1_1k1999_th_5-180-200.pkl mean AP  0.1335
# v6_3_miss60_minscore0p3VidORtrain_freq2to1_5k5999_th_5-180-200.pkl  mean AP  0.2043

## ----------- MEGA inference freq1 -----------
# v6_3_miss60_minscore0p3VidORval_freq1_th_15-180-200-0.40.pkl mean AP  0.1167
# v6_3_miss60_minscore0p3VidORval_freq1_th_15-180-200-0.40.pkl (with enti_cls from encoder) mean AP  0.1141   see dis_eval_or_enticls.py
# v6_3_miss60_minscore0p3VidORval_freq1_th_15-180-200-0.40.pkl (only consider bbox postion) mean AP  0.2308

# /home/gkf/project/vidvrd-mff/ORval_traj.pkl  mean AP  0.0863  (vidvrd-helper, mean AP  0.0882)
# /home/gkf/project/vidvrd-mff/ORval_traj_FG.pkl (only consider bbox postion) mean AP 0.1664   (vidvrd-helper, has not wrote the code yet.)