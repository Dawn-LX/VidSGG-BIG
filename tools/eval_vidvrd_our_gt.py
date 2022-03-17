
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
    infer_result = {}
    infer_result_for_save = {}
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal_list = [p.to(device) for p in proposal_list]
        gt_graph_list = [g.to(device) for g in gt_graph_list]
        with torch.no_grad():
            batch_triplets = model(proposal_list,topk=topk)

        assert len(proposal_list) == 1
        video_name = proposal_list[0].video_name
        if batch_triplets[0] is None:
            infer_result.update({video_name:None})
            continue

        (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        ) = batch_triplets[0]
        uniq_scores = torch.mean(uniq_scores,dim=-1)   # (n_unique,)
        # print(uniq_scores.shape)
        results = (uniq_quintuples.cpu(),uniq_scores.cpu(),uniq_dura_inters.cpu())
        infer_result.update(
            {video_name:results}
        )
        res = [x.cpu() for x in batch_triplets[0]]
        infer_result_for_save[video_name] = res

    if save_infer_result:
        save_path = os.path.join(experiment_dir,'VidVRDtest_infer_result_{}.pkl'.format(save_tag))
        with open(save_path,'wb') as f:
            pickle.dump(infer_result_for_save,f)
        logger.info("infer_result saved at {}".format(save_path))

    logger.info("start convert format for evaluate...")


    convertor_all = EvalFmtCvtor("vidvrd")
    
    gt_relations = {}
    predict_relations = {}
    for proposal_list,gt_graph_list in tqdm(dataloader):
        assert len(proposal_list) == 1
        proposal = proposal_list[0]
        gt_graph = gt_graph_list[0]

        pr_triplet = infer_result[proposal.video_name]
        
        pr_result = convertor_all.to_eval_format_pr(proposal,pr_triplet,use_pku=use_pku)
        predict_relations.update(pr_result) 
        gt_result = convertor_all.to_eval_format_gt(gt_graph)
        gt_relations.update(gt_result)
    

    logger.info('Computing average precision AP over {} videos...'.format(len(gt_relations)))
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,predict_relations,viou_threshold=0.5)
    

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


def replace_state_dict_keys(state_dict):
    """
    This func is for debug
    """
    state_dict_new = {}
    for name,v in state_dict.items():
        if name == "pred_nodes_init":
            name = "pred_query_init"
        
        if "fc_msg_recv" in name:
            name = name.replace("fc_msg_recv","fc_rolewise")
        
        if ".layers." in name:
            name = name.replace(".layers.",".")   # MLP --> nn.Sequential(...)
        
        if "fc_pred2logits.0" in name:
            name = name.replace("fc_pred2logits.0","fc_pred2logits")

        state_dict_new[name] = v


    return state_dict_new


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
    ### exp1
    python tools/eval_vidvrd_our_gt.py \
        --cfg_path experiments/exp1/config_.py \
        --ckpt_path /home/gkf/project/VideoGraph/training_dir_reorganized/vidvrd/model_0v10_cachePKUv1_rightcatid/model_epoch_80.pth \
        --use_pku \
        --cuda 1 \
        --save_tag debug
    
    ### exp2
    python tools/eval_vidvrd_our_gt.py \
        --cfg_path experiments/exp2/config_.py \
        --ckpt_path /home/gkf/project/VideoGraph/training_dir_reorganized/vidvrd/model_0v10_pku_i3dclsme2_cachePKUv2/model_epoch_70.pth \
        --use_pku \
        --cuda 2 \
        --save_tag debug
    
    ### exp3
    python tools/eval_vidvrd_our_gt.py \
        --cfg_path experiments/exp3/config_.py \
        --ckpt_path /home/gkf/project/VidSGG-BIG/experiments/exp3/model_epoch_80.pth \
        --cuda 3 \
        --save_tag debug
    
    '''

    ##### ablation for PKU RoI+I3D
    # cfg_path = "training_dir_reorganized/vidvrd/model_0v10_pku_i3d_cachePKUv2/config_.py"
    # model_class_path = "Tempformer_model/model_0v10_pku_i3d.py"
    # # model_class_path = "Tempformer_model/model_vrd02.py"
    # train(model_class_path,cfg_path,device_ids=[1,2],from_checkpoint=False,ckpt_path=None)

    # cfg_path = "training_dir_reorganized/vidvrd/model_0v10_cachePKUv1_rightcatid/config_.py"
    # model_class_path = "Tempformer_model/model_0v10.py"
    # weight_path = "training_dir_reorganized/vidvrd/model_0v10_cachePKUv1_rightcatid/model_epoch_80.pth"
    # train(model_class_path,cfg_path,device_ids=[0],from_checkpoint=False,ckpt_path=None)
