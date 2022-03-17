
import root_path
import pickle
import json
from tqdm import tqdm
import os
import argparse
import torch

from models import BIG_C_vidor,Base_C
from dataloaders.dataloader_vidor_v3 import Dataset
# from dataloaders.dataloader_vidor import Dataset
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import create_logger,parse_config_py
from VidVRDhelperEvalAPIs import eval_visual_relation

torch.set_printoptions(sci_mode=False,precision=4,linewidth=160)

def evaluate_cls_stage(
    model_class,
    cfg_path,
    weight_path,
    save_tag="",
    use_regr = None,
    experiment_dir=None,
    gpu_id = 0,
    save_infer_result=False,
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

    model = model_class(model_config,is_train=False)

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
    infer_result = {}
    infer_result_for_save = {}
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal_list = [p.to(device) for p in proposal_list]
        gt_graph_list = [g.to(device) for g in gt_graph_list]
        with torch.no_grad():
            batch_triplets = model(proposal_list,use_post=use_post,topk=topk,use_regr=use_regr)

        assert len(proposal_list) == 1
        video_name = proposal_list[0].video_name
        if batch_triplets[0] is None:
            infer_result.update({video_name:None})
            infer_result_for_save[video_name] = None
            continue

        (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        ) = batch_triplets[0]
        uniq_scores = torch.mean(uniq_scores,dim=-1)   # (n_unique,)
        results = (uniq_quintuples.cpu(),uniq_scores.cpu(),uniq_dura_inters.cpu())
        infer_result.update(
            {video_name:results}
        )
        res = [x.cpu() for x in batch_triplets[0]]
        infer_result_for_save[video_name] = res

    if save_infer_result:
        save_path = os.path.join(experiment_dir,'VidORval_infer_result_{}.pkl'.format(save_tag))
        with open(save_path,'wb') as f:
            pickle.dump(infer_result_for_save,f)
        logger.info("infer_result saved at {}".format(save_path))

    logger.info("start convert format for evaluate...")
    
    convertor_all = EvalFmtCvtor("vidor")
    gt_relations = {}
    predict_relations = {}

    for proposal_list,gt_graph_list in tqdm(dataloader):
        assert len(proposal_list) == 1
        proposal = proposal_list[0]
        gt_graph = gt_graph_list[0]

        pr_triplet = infer_result[proposal.video_name]
        

        pr_result = convertor_all.to_eval_format_pr(proposal,pr_triplet,preserve_debug_info=True)
        predict_relations.update(pr_result) 
        gt_result = convertor_all.to_eval_format_gt(gt_graph)
        gt_relations.update(gt_result)
    

    logger.info('Computing average precision AP over {} videos...'.format(len(gt_relations)))
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,predict_relations,viou_threshold=0.5,det_nreturns=[50,100])

    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))

    if save_relation_json:
        save_path = os.path.join(experiment_dir,'VidORval_predict_relations_{}.json'.format(save_tag))
        logger.info("saving predict_relations into {}...".format(save_path))
        with open(save_path,'w') as f:
            json.dump(predict_relations,f)
        logger.info("predict_relations have been saved at {}".format(save_path))
        
    logger.info("log file have been saved at {}".format(log_path))
    logger.handlers.clear()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--cuda", type=int,help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--use_baseline", action="store_true",default=False,help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--json_results_path", type=str,help="...")
    args = parser.parse_args()

    if args.use_baseline:
        model_class = Base_C
    else:
        model_class = BIG_C_vidor
    
    evaluate_cls_stage(
        model_class,
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
    python tools/eval_vidor_our_gt.py \
        --cfg_path experiments/exp4/config_.py \
        --ckpt_path experiments/exp4/model_epoch_60.pth \
        --save_tag epoch60_debug \
        --cuda 1

    2022-03-09 22:58:18,014 - detection mean AP (used in challenge): 0.0803040571943109
    2022-03-09 22:58:18,016 - detection recall: {50: 0.076040074, 100: 0.09395528}
    2022-03-09 22:58:18,016 - tagging precision: {1: 0.6225961538461539, 5: 0.5096153934987692, 10: 0.4030048120766878}
    2022-03-09 22:58:18,016 - evaluate results and log file have been saved at experiments/exp4/logfile/eval_k3_rTrue_pTrue_eepoch60_debug.log
    '''


    '''
    ### table-3 BIG-C RoI + Lang
    python tools/eval_vidor_our_gt.py \
        --cfg_path experiments/exp5/config_.py \
        --ckpt_path experiments/exp5/model_epoch_60.pth \
        --save_tag epoch60_debug \
        --cuda 1
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

    '''
    ### table-2 Base-C
    python tools/eval_vidor_our_gt.py \
        --use_baseline \
        --cfg_path experiments/exp6/config_.py \
        --ckpt_path experiments/exp6/model_epoch_80.pth \
        --save_tag epoch80_rt200 \
        --cuda 1
    2022-03-14 04:02:14,310 - detection mean AP (used in challenge): 0.06984874101064151
    2022-03-14 04:02:14,311 - detection recall: {50: 0.071760334, 100: 0.09199788}
    2022-03-14 04:02:14,311 - tagging precision: {1: 0.5913461538461539, 5: 0.4704927962918121, 10: 0.3807534959095602}
    2022-03-14 04:02:14,311 - log file have been saved at experiments/exp6/logfile/eval_k3_rTrue_pTrue_eepoch80_rt200.log
    '''


    
    