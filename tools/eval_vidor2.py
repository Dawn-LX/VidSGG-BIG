
import root_path
import pickle
import json
from tqdm import tqdm
import os
import argparse
import torch

from models import BIG_C_vidor,DEBUG
from dataloaders.dataloader_vidor_v3 import Dataset
# from dataloaders.dataloader_vidor import Dataset
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import create_logger,parse_config_py
from VidVRDhelperEvalAPIs import eval_relation_with_gt

torch.set_printoptions(sci_mode=False,precision=4,linewidth=160)

def evaluate_cls_stage(
    cfg_path,
    weight_path,
    save_tag="",
    experiment_dir=None,
    gpu_id = 0,
    save_infer_result = True,
    save_relation_json=False
):

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
    model_config = all_cfgs["model_config"]
    infer_config = all_cfgs["inference_config"]
    topk=infer_config["topk"]

    save_tag = "topk{}_{}".format(topk,save_tag)

    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(infer_config)

    model = BIG_C_vidor(model_config,is_train=False)

    model = model.cuda(device)
    state_dict = torch.load(weight_path,map_location=torch.device('cpu'))
    state_dict = state_dict["model_state_dict"]
    temp = next(iter(state_dict.keys()))
    if temp.startswith("module."):  # for model's state_dict saved after multi-gpu training
        state_dict_ = {}
        for k in state_dict.keys():
            state_dict_[k[7:]] = state_dict[k]
        model.load_state_dict(state_dict_)
    else:
        model.load_state_dict(state_dict)
    model.eval()


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
    logger.info("weight_path:{}".format(weight_path))

    convertor = EvalFmtCvtor("vidor")
    predict_relations = dict()
    infer_result_for_save = dict()
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal_list = [p.to(device) for p in proposal_list]
        gt_graph_list = [g.to(device) for g in gt_graph_list]
        with torch.no_grad():
            batch_triplets = model(proposal_list,topk=topk)

        
        assert len(proposal_list) == 1
        proposal = proposal_list[0].to(torch.device("cpu"))
        video_name = proposal_list[0].video_name

        if batch_triplets[0] is None:
            infer_result = None
            infer_result_for_save[video_name] = None
            continue

        (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        ) = batch_triplets[0]
        uniq_scores = torch.mean(uniq_scores,dim=-1)   # (n_unique,)
        infer_result = (uniq_quintuples.cpu(),uniq_scores.cpu(),uniq_dura_inters.cpu())
        infer_result_for_save[video_name] = [x.cpu() for x in batch_triplets[0]]

        pr_result = convertor.to_eval_format_pr(proposal,infer_result,preserve_debug_info=True)
        predict_relations.update(pr_result) 

    eval_relation_with_gt(
        dataset_type="vidor",
        logger=logger,
        prediction_results=predict_relations
    )
    if save_infer_result:
        save_path =  os.path.join(experiment_dir,'VidORval_infer_results_{}.pkl'.format(save_tag))
        logger.info("saving infer_results into {}...".format(save_path))
        with open(save_path,'wb') as f:
            pickle.dump(infer_result_for_save,f)
        logger.info("infer_results have been saved at {}".format(save_path))
    if save_relation_json:
        save_path = os.path.join(experiment_dir,'VidORval_predict_relations_{}.json'.format(save_tag))
        logger.info("saving predict_relations into {}...".format(save_path))
        with open(save_path,'w') as f:
            json.dump(predict_relations,f)
        logger.info("predict_relations have been saved at {}".format(save_path))
    
    logger.info("log file have been saved at {}".format(log_path))
    logger.handlers.clear()
    


def evaluate_combined(
    cfg_path,
    weight_path,
    cls_stage_result_path,
    save_tag="",
    experiment_dir=None,
    gpu_id = 0,
    save_infer_result = True,
    save_relation_json=False
):

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
    model_config = all_cfgs["model_config"]

    infer_config = all_cfgs["inference_config"]
    score_th = infer_config["score_th"]
    tiou_th = infer_config["tiou_th"]
    slots_th = infer_config["slots_th"]
    nms_th = infer_config["nms_th"]

    # create logger
    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(infer_config)

    model = DEBUG(model_config,is_train=False)
    

    model = model.cuda(device)
    state_dict = torch.load(weight_path,map_location=torch.device('cpu'))
    state_dict = state_dict["model_state_dict"]
    temp = next(iter(state_dict.keys()))
    if temp.startswith("module."):  # for model's state_dict saved after multi-gpu training
        state_dict_ = {}
        for k in state_dict.keys():
            state_dict_[k[7:]] = state_dict[k]
        model.load_state_dict(state_dict_)
    else:
        model.load_state_dict(state_dict)
    model.eval()

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
    logger.info("weight_path:{}".format(weight_path))

    with open(cls_stage_result_path,'rb') as f:
        infer_result_cls_stage = pickle.load(f)
    logger.info("infer_results of cls_stage loaded from {}".format(cls_stage_result_path))

    predict_relations ={}
    convertor = EvalFmtCvtor("vidor")
    for video_feature_list, proposal_list,gt_graph_list in tqdm(dataloader):
        video_feature_list = [v.to(device) for v in video_feature_list]
        proposal_list = [p.to(device) for p in proposal_list]
        assert len(proposal_list) == 1
        proposal = proposal_list[0].to(torch.device("cpu"))
        video_name = proposal.video_name
        video_len = proposal.video_len

        batch_triplets = infer_result_cls_stage[video_name]        
        if batch_triplets is None:
            infer_result = None
            continue
        
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
            predictions,slots_probs,slots_mask = model(video_feature_list,data_list,score_th,tiou_th,slots_th,nms_th,truncated=True,with_gt_data=False)
            # (n_uniq,n_slots+1,2)  (n_uniq,n_slots+1),(n_uniq,n_slots+1)
        
        n_uniq,n_slots = slots_probs.shape
        uniq_quintuples = uniq_quintuples[:,None,:].repeat(1,n_slots,1)      # (n_uniq,n_slots,5)
        uniq_scores = uniq_scores[:,None] * slots_probs   # (n_unique,n_slots)
        predictions = predictions * video_len                                # (n_uniq,n_slots,2)
        
        uniq_quintuples = uniq_quintuples[slots_mask,:].cpu()
        uniq_scores = uniq_scores[slots_mask].cpu()
        predictions = predictions[slots_mask,:].cpu()
        predictions = torch.round(predictions).type(torch.long)
        
        infer_result = (uniq_quintuples,uniq_scores,predictions)
        pr_result = convertor.to_eval_format_pr(proposal,infer_result)
        predict_relations.update(pr_result) 

    logger.info('Computing average precision AP over {} videos...'.format(len(gt_relations)))
    mean_ap, rec_at_n, mprec_at_n = eval_relation_with_gt(
        dataset_type="vidor",
        logger=logger,
        prediction_results=predict_relations
    )

    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))
        

    logger.info("log file have been saved at {}".format(log_path))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--cuda", type=int,help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--json_results_path", type=str,help="...")
    args = parser.parse_args()


    if args.json_results_path is not None:
        eval_relation_with_gt(
            dataset_type="vidor",
            json_results_path=args.json_results_path
        )
    else:
        evaluate_cls_stage(
            args.cfg_path,
            args.ckpt_path,
            save_tag=args.save_tag,
            experiment_dir=args.output_dir,
            gpu_id = args.cuda,
            save_infer_result=True,
            save_relation_json=False
        )
    



    '''
    ### table-3 BIG-C RoI
    python tools/eval_vidor2.py \
        --cfg_path experiments/exp4/config_.py \
        --ckpt_path experiments/exp4/model_epoch_60.pth \
        --save_tag epoch60_debug \
        --cuda 1
    022-03-10 15:54:44,476 - detection mean AP (used in challenge): 0.08030405719431091
    2022-03-10 15:54:44,477 - detection recall: {50: 0.076040074, 100: 0.09395528}
    2022-03-10 15:54:44,477 - tagging precision: {1: 0.6225961538461539, 5: 0.5096153934987692, 10: 0.4030048120766878}
    2022-03-10 15:54:45,678 - log file have been saved at experiments/exp4/logfile/eval_k3_rTrue_pTrue_eepoch60_debug.log
    '''


    '''
    ### table-3 BIG-C RoI + Lang
    python tools/eval_vidor2.py \
        --cfg_path experiments/exp5/config_.py \
        --ckpt_path experiments/exp5/model_epoch_60.pth \
        --save_tag epoch60_debug \
        --cuda 2
    
    2022-03-10 15:15:34,062 - detection mean AP (used in challenge): 0.08296007380280078
    2022-03-10 15:15:34,064 - detection recall: {50: 0.079225, 100: 0.09657621}
    2022-03-10 15:15:34,064 - tagging precision: {1: 0.6442307692307693, 5: 0.5170673167404647, 10: 0.41052017033171767}
    2022-03-10 15:15:35,310 - log file have been saved at experiments/exp5/logfile/eval_k3_rTrue_pTrue_eepoch60_debug.log

    '''

    ### table-3 BIG-C RoI + Lang
    # model_class_path = "Tempformer_model/model_0v7.py"
    # cfg_path = "training_dir_reorganized/vidor/model_0v7/config_.py"
    # weight_path = "training_dir_reorganized/vidor/model_0v7/model_epoch_60.pth"
    # save_tag = "epoch60_debug"
    # 2022-03-09 21:04:59,378 - detection mean AP (used in challenge): 0.08296007380280078
    # 2022-03-09 21:04:59,379 - detection recall: {50: 0.079225, 100: 0.09657621}
    # 2022-03-09 21:04:59,379 - tagging precision: {1: 0.6442307692307693, 5: 0.5170673167404647, 10: 0.41052017033171767}
    # 2022-03-09 21:04:59,390 - hit_infos have been saved at training_dir_reorganized/vidor/model_0v7/VidORval_hit_infos_k3_rFalse_pTrue_eepoch60_debug.pkl
    # 2022-03-09 21:04:59,391 - evaluate results and log file have been saved at training_dir_reorganized/vidor/model_0v7/logfile/eval_k3_rFalse_pTrue_eepoch60_debug.log


    
    