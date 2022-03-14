
import pickle

import root_path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter       
from tqdm import tqdm
from collections import defaultdict


from importlib import import_module
from dataloaders.dataloader_vidor_v3 import Dataset
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import create_logger,parse_config_py
from utils.utils_func import unique_with_idx_nd
from VidVRDhelperEvalAPIs import eval_visual_relation

torch.set_printoptions(sci_mode=False,precision=4,linewidth=160)

def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size

def save_checkpoint(batch_size,crt_epoch,model,optimizer,scheduler,save_path):
    checkpoint = {
        "batch_size":batch_size,
        "crt_epoch":crt_epoch + 1,
        "model_state_dict":model.state_dict(),
        "optim_state_dict":optimizer.state_dict(),
        "sched_state_dict":scheduler.state_dict(),
    }
    torch.save(checkpoint,save_path)

 

def eval_combined_improve_from_file(model2_class_path,weight2_path,infer_result_path,
    cfg_path,save_tag="",experiment_dir=None,gpu_id = 0,max_slots_prob_th=0.2,save_relation_result=False):
    
    ## import model class 
    temp = model2_class_path.split('.')[0].split('/')
    model2_class_path = ".".join(temp)
    DEBUG = import_module(model2_class_path).DEBUG


    device = torch.device("cuda:{}".format(gpu_id))
    
    ## create dirs
    if experiment_dir == None:
        experiment_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(experiment_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    ## load configs
    all_cfgs = parse_config_py(cfg_path)
    dataset_config = all_cfgs["test_dataset_config"]
    model2_config = all_cfgs["model_config"]

    infer_config = all_cfgs["inference_config"]
    score_th = infer_config["score_th"][0]
    tiou_th = infer_config["tiou_th"][0]
    slots_th = infer_config["slots_ths"][0]
    nms_th = infer_config["nms_ths"][0]

    # create logger

    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model2_config)
    logger.info(dataset_config)
    logger.info(infer_config)

    model2 = DEBUG(model2_config,is_train=False)
    
    model2 = model2.cuda(device)
    state_dict = torch.load(weight2_path,map_location=torch.device('cpu'))
    state_dict = state_dict["model_state_dict"]
    state_dict_ = {}
    for k in state_dict.keys():
        state_dict_[k[7:]] = state_dict[k]
    model2.load_state_dict(state_dict_)
    model2.eval()

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

    logger.info("start inference...")
    logger.info("infer_config:{}".format(infer_config))
    logger.info("weight2_path:{}".format(weight2_path))

    with open(infer_result_path,'rb') as f:
        infer_result_cls_stage = pickle.load(f)
    logger.info("infer_result_path loaded from {}".format(infer_result_path))
    infer_result = {}
    for video_feature_list, proposal_list,gt_graph_list in tqdm(dataloader):
        video_feature_list = [v.to(device) for v in video_feature_list]
        proposal_list = [p.to(device) for p in proposal_list]
        assert len(proposal_list) == 1
        video_name = proposal_list[0].video_name
        video_len = proposal_list[0].video_len

        batch_triplets = infer_result_cls_stage[video_name]

        
        if batch_triplets is None:
            infer_result.update({video_name:None})
            continue
        
        # for x in batch_triplets:
        #     print(x.shape)
        (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        ) = batch_triplets
        uniq_scores = torch.mean(uniq_scores,dim=-1)
        uniq_scores = uniq_scores.to(device)
        datas = (uniq_quintuples.to(device),uniq_dura_inters.to(device),video_len)
        data_list = [datas]
        with torch.no_grad():
            predictions,slots_probs,slots_mask = model2(video_feature_list,data_list,score_th,tiou_th,slots_th,nms_th,truncated=True,with_gt_data=False)
            # (n_uniq,n_slots+1,2)  (n_uniq,n_slots+1),(n_uniq,n_slots+1)
        
        n_uniq,n_slots = slots_probs.shape



        uniq_quintuples = uniq_quintuples[:,None,:].repeat(1,n_slots,1)      # (n_uniq,n_slots,5)
        uniq_scores = uniq_scores[:,None] * slots_probs   # (n_unique,n_slots)
        # uniq_scores = (torch.sqrt(uniq_scores[:,0,None] * slots_probs) + uniq_scores[:,1:].sum(dim=-1)[:,None])/3
        predictions = predictions * video_len                                # (n_uniq,n_slots,2)
        
        uniq_quintuples = uniq_quintuples[slots_mask,:].cpu()
        uniq_scores = uniq_scores[slots_mask].cpu()
        predictions = predictions[slots_mask,:].cpu()
        predictions = torch.round(predictions).type(torch.long)
        # print(predictions,predictions.shape)
        
        infer_result.update(
            {video_name:(uniq_quintuples,uniq_scores,predictions)}
        )
        # break
        del video_feature_list,proposal_list,gt_graph_list,data_list

    logger.info("start convert format for evaluate...")
    
    # convertor_sep = EvalFmtCvtor("vidor",eval_separately=True)
    convertor_all = EvalFmtCvtor("vidor")
    # predict_relations = {name:{} for name in convertor_sep.part_names}
    # gt_relations = {name:{} for name in convertor_sep.part_names}
    predict_relations ={}
    gt_relations = {}

    for _,proposal_list,gt_graph_list in tqdm(dataloader):
        assert len(proposal_list) == 1
        proposal = proposal_list[0]
        gt_graph = gt_graph_list[0]

        pr_triplet = infer_result[proposal.video_name]
        
        
        # for eval overall
        pr_result = convertor_all.to_eval_format_pr(proposal,pr_triplet)
        predict_relations.update(pr_result) 
        gt_result = convertor_all.to_eval_format_gt(gt_graph)
        gt_relations.update(gt_result)

    
    logger.info('Computing average precision AP over {} videos...'.format(len(gt_relations)))
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,predict_relations,viou_threshold=0.5,det_nreturns=[50,100,200,250,1000])

    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))
        

    logger.info("log file have been saved at {}".format(log_path))



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

def eval_recall_for_each_part(cfg_path,hit_info_path,experiment_dir=None):
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
        gt2hit_ids = torch.from_numpy(gt2hit_ids)

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



if __name__ == "__main__":
    

    

    cfg_path = "experiments/grounding_weights/config_.py"
    model2_class_path = "models/groundOnly_5.py"
    weight2_path = "experiments/grounding_weights/model_epoch_70.pth"
  
    ### table-3 BIG RoI
    infer_result_path = "experiments/exp4/VidORval_infer_results_topk3_epoch60_debug.pkl"
    save_tag = "GO5slot10_e70_mean_mul_nms0.8_imp_exp4"
    # 2022-03-11 10:43:53,557 - Computing average precision AP over 835 videos...
    # 2022-03-11 10:46:37,949 - detection mean AP (used in challenge): 0.0828240818473417
    # 2022-03-11 10:46:37,950 - detection recall: {50: 0.077400304, 100: 0.09820184, 200: 0.11654834, 250: 0.12232102, 1000: 0.13721718}
    # 2022-03-11 10:46:37,950 - tagging precision: {1: 0.6213942307692307, 5: 0.5125000089621887, 10: 0.40480769666520733}

    ### table-3 BIG RoI + lang
    # infer_result_path = "experiments/exp5/VidORval_infer_results_topk3_epoch60_debug.pkl"
    # save_tag = "GO5slot10_e70_mean_mul_nms0.8_imp"
    # 2022-03-11 10:19:25,525 - Computing average precision AP over 835 videos...
    # 2022-03-11 10:22:06,567 - detection mean AP (used in challenge): 0.08545435481766761
    # 2022-03-11 10:22:06,568 - detection recall: {50: 0.08038617, 100: 0.100424655, 200: 0.11970009, 250: 0.12550594, 1000: 0.1410988}
    # 2022-03-11 10:22:06,569 - tagging precision: {1: 0.6442307692307693, 5: 0.5180288552163312, 10: 0.4096788243086149}


    eval_combined_improve_from_file(model2_class_path,weight2_path,infer_result_path,cfg_path,save_tag=save_tag,gpu_id = 3)

   