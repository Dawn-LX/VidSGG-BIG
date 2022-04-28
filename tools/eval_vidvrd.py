

import root_path

import pickle
import json
import os
import argparse
from tqdm import tqdm
import torch

from dataloaders.dataloader_vidvrd import Dataset,Dataset_pku,Dataset_pku_i3d
from models import BIG_C_vidvrd
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import create_logger,parse_config_py
from VidVRDhelperEvalAPIs import eval_relation_with_gt

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


def inference_then_eval(
    cfg_path,
    weight_path,
    save_tag="",
    use_pku=False,
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
    topk=infer_config["topk"]

    # create logger
    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(infer_config)

    model = BIG_C_vidvrd(model_config,is_train=False)

    model = model.cuda(device)
    state_dict = torch.load(weight_path,map_location=torch.device('cpu'))
    state_dict = state_dict["model_state_dict"]
    state_dict_ = {}
    for k in state_dict.keys():
        state_dict_[k[7:]] = state_dict[k]
    # state_dict_ = replace_state_dict_keys(state_dict_)  ## for debug
    model.load_state_dict(state_dict_)
    model.eval()


    ## construct dataset
    if use_pku:
        use_i3d = dataset_config.get("i3d_dir",None) is not None
        if use_i3d:
            dataset = Dataset_pku_i3d(**dataset_config)
        else:
            dataset = Dataset_pku(**dataset_config)        
    else:
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

    convertor = EvalFmtCvtor("vidvrd")
    predict_relations = {}
    infer_result_for_save = {}
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
            continue

        (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        ) = batch_triplets[0]
        uniq_scores = torch.mean(uniq_scores,dim=-1)   # (n_unique,)
        infer_result = (uniq_quintuples.cpu(),uniq_scores.cpu(),uniq_dura_inters.cpu())
        infer_result_for_save[video_name] = [x.cpu() for x in batch_triplets[0]]  # for debug

        pr_result = convertor.to_eval_format_pr(proposal,infer_result,use_pku=use_pku)
        predict_relations.update(pr_result)

    if save_infer_result:
        save_path = os.path.join(experiment_dir,'VidVRDtest_infer_result_{}.pkl'.format(save_tag))
        with open(save_path,'wb') as f:
            pickle.dump(infer_result_for_save,f)
        logger.info("infer_result saved at {}".format(save_path))

    eval_relation_with_gt(
        dataset_type="vidvrd",
        logger=logger,
        prediction_results=predict_relations
    )
    
    if save_relation_json:
        save_path = os.path.join(experiment_dir,'VidVRDtest_predict_relations_{}.json'.format(save_tag))
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
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--json_results_path", type=str,help="...")
    parser.add_argument("--use_pku",action="store_true",default=False,help="...")
    args = parser.parse_args()


    if args.json_results_path is not None:
        eval_relation_with_gt(
            dataset_type="vidvrd",
            json_results_path=args.json_results_path
        )
    else:
        inference_then_eval(
        args.cfg_path,
        args.ckpt_path,
        save_tag=args.save_tag,
        use_pku=args.use_pku,
        experiment_dir=args.output_dir,
        gpu_id = args.cuda,
        save_infer_result=False,
        save_relation_json=False
        )

    '''
    python tools/eval_vidvrd.py \
        --json_results_path /home/gkf/project/VideoGraph/training_dir_reorganized/vidvrd/model_0v10_pku_i3dclsme2_cachePKUv2/VidORval_predict_relations_topk10_pTrue_epoch70.json
    
    ### exp1
    python tools/eval_vidvrd.py \
        --cfg_path experiments/exp1/config_.py \
        --ckpt_path experiments/exp1/model_epoch_80.pth \
        --use_pku \
        --cuda 1 \
        --save_tag debug
    
    2022-03-09 02:41:18,550 - detection mean AP (used in challenge): 0.1756102305112229
    2022-03-09 02:41:18,551 - detection recall: {50: 0.095966905, 100: 0.109203726}
    2022-03-09 02:41:18,551 - tagging precision: {1: 0.565, 5: 0.4430000091344118, 10: 0.32350000478327273}
    2022-03-09 02:41:18,668 - log file have been saved at experiments/exp1/logfile/eval_debug.log
    
    
    ### exp2
    python tools/eval_vidvrd.py \
        --cfg_path experiments/exp2/config_.py \
        --ckpt_path experiments/exp2/model_epoch_70.pth \
        --use_pku \
        --cuda 2 \
        --save_tag debug
    
    2022-03-09 02:39:43,260 - detection mean AP (used in challenge): 0.17679591933675462
    2022-03-09 02:39:43,261 - detection recall: {50: 0.09638056, 100: 0.11292658}
    2022-03-09 02:39:43,261 - tagging precision: {1: 0.56, 5: 0.438000009059906, 10: 0.32850000374019145}

    
    ### exp3
    python tools/eval_vidvrd.py \
        --cfg_path experiments/exp3/config_.py \
        --ckpt_path experiments/exp3/model_epoch_80.pth \
        --cuda 3 \
        --save_tag debug
    
    2022-03-09 02:41:30,289 - detection mean AP (used in challenge): 0.26088200440572606
    2022-03-09 02:41:30,289 - detection recall: {50: 0.14105481, 100: 0.16256464}
    2022-03-09 02:41:30,289 - tagging precision: {1: 0.73, 5: 0.551, 10: 0.4}
    2022-03-09 02:41:30,400 - log file have been saved at experiments/exp3/logfile/eval_debug.log
    '''