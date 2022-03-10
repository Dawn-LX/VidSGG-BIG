
import root_path
import pickle
import json
from tqdm import tqdm
import os
import argparse
import torch

from models import BIG_C_vidor
from dataloaders.dataloader_vidor_v3 import Dataset
# from dataloaders.dataloader_vidor import Dataset
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import create_logger,parse_config_py,load_json
from VidVRDhelperEvalAPIs import eval_visual_relation

torch.set_printoptions(sci_mode=False,precision=4,linewidth=160)

def inference_then_eval(
    cfg_path,
    weight_path,
    save_tag="",
    use_regr = None,
    experiment_dir=None,
    gpu_id = 0,
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
    if use_regr is None:
        use_regr = infer_config["use_regr"]
    else:
        infer_config["use_regr"] = use_regr
    use_post=infer_config["use_post"]
    topk=infer_config["topk"]

    # create logger
    # save_tag = "topk-{}_regr-{}_postpro-{}_epoch-{}".format(topk,use_regr,use_post,epoch) # this filename is too long
    save_tag = "k{}_r{}_p{}_e{}".format(topk,use_regr,use_post,save_tag)

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

    convertor_all = EvalFmtCvtor("vidor")
    predict_relations = dict()
    infer_result_for_save = dict()
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal_list = [p.to(device) for p in proposal_list]
        gt_graph_list = [g.to(device) for g in gt_graph_list]
        with torch.no_grad():
            batch_triplets = model(proposal_list,use_post=use_post,topk=topk,use_regr=use_regr)

        
        assert len(proposal_list) == 1
        proposal = proposal_list[0].to(torch.device("cpu"))
        video_name = proposal_list[0].video_name

        if batch_triplets[0] is None:
            infer_result = None
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

        pr_result = convertor_all.to_eval_format_pr(proposal,infer_result,preserve_debug_info=True)
        predict_relations.update(pr_result) 

    eval_relation(logger=logger,prediction_results=predict_relations)
    
    if save_relation_json:
        save_path = os.path.join(experiment_dir,'VidORval_predict_relations_{}.json'.format(save_tag))
        logger.info("saving predict_relations into {}...".format(save_path))
        with open(save_path,'w') as f:
            json.dump(predict_relations,f)
        logger.info("predict_relations have been saved at {}".format(save_path))
    logger.info("log file have been saved at {}".format(log_path))
    

 
def eval_relation(
    logger=None,
    prediction_results=None,
    json_results_path=None,
):
    '''
    NOTE this func is only support for VidVRD currently
    '''
    if logger is None:
        log_dir = "cache/logfile"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir,'eval.log')
        logger = create_logger(log_path)
    if prediction_results is None:
        logger.info("loading json results from {}".format(json_results_path))
        prediction_results = load_json(json_results_path)
        logger.info("Done.")
    else:
        assert json_results_path is None
    

    gt_relations = load_json("datasets/GT_json_for_eval/VidORval_gts.json")
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,prediction_results,viou_threshold=0.5)
    # logger.info(f"mAP:{mean_ap}, Retection Recall:{rec_at_n}, Tagging Precision: {mprec_at_n}")
    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--cuda", type=int,help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--json_results_path", type=str,help="...")
    parser.add_argument("--use_pku",action="store_true",default=False,help="...")
    args = parser.parse_args()


    if args.json_results_path is not None:
        eval_relation(
            json_results_path=args.json_results_path,
        )
    else:
        inference_then_eval(
            args.cfg_path,
            args.ckpt_path,
            save_tag=args.save_tag,
            experiment_dir=args.output_dir,
            gpu_id = args.cuda,
            save_relation_json=False
        )
    


    # inference_then_eval(cfg_path,weight_path,save_tag,use_regr=False,gpu_id=3,save_relation_result=False)

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
        --cuda 1
    
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


    
    