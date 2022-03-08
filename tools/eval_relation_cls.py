import root_path
import argparse
import os 
import json
from tqdm import tqdm
from collections import defaultdict

from utils.utils_func import create_logger
from VidVRDhelperEvalAPIs import eval_visual_relation

def load_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x

def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size




def inference_then_eval(
    cfg_path,
    ckpt_path_traj,
    ckpt_path_pred,
    output_dir=None,
    device_id=0,
    eval_split_traj="all",
    eval_split_pred="novel",
    save_tag="",
    save_json_results=False
):

    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    logger = create_logger(log_path)


    configs = parse_config_py(cfg_path)
    dataset_cfg = configs["eval_dataset_cfg"]
    model_traj_cfg = configs["model_traj_cfg"]
    model_pred_cfg = configs["model_pred_cfg"]
    eval_cfg = configs["eval_cfg"]
    device = torch.device(f"cuda:{device_id}")
        

    logger.info("dataset config: {}".format(dataset_cfg))
    logger.info("model_traj config: {}".format(model_traj_cfg))
    logger.info("model_pred config: {}".format(model_pred_cfg))
    logger.info("evaluate config: {}".format(eval_cfg))

    


    model_traj = OpenVocTrajCls(model_traj_cfg,is_tran=False)
    logger.info(f"loading check point from {ckpt_path_traj}")
    check_point = torch.load(ckpt_path_traj,map_location=torch.device('cpu'))
    state_dict = check_point["model_state_dict"]
    model_traj = model_traj.to(device)
    state_dict = modify_state_dict(state_dict)
    model_traj.load_state_dict(state_dict)
    model_traj.eval()
    model_traj.reset_classifier_weights(eval_split_traj)

    model_pred = OpenVocRelationCls(model_pred_cfg,is_tran=False)
    logger.info(f"loading check point from {ckpt_path_pred}")
    check_point = torch.load(ckpt_path_pred,map_location=torch.device("cpu"))
    state_dict = check_point["model_state_dict"]
    model_pred = model_pred.to(device)
    model_pred.load_state_dict(state_dict)
    model_pred.eval()
    model_pred.reset_classifier_weights(eval_split_pred)

    logger.info("preparing dataloader...")
    dataset = VidVRDTripletDataset(**dataset_cfg)
    collate_func = dataset.get_collator_func()
    to_device_func = get_to_device_func(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = collate_func ,
        num_workers = 2,
        drop_last= False,
        shuffle= False,
    )
    logger.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    logger.info(
        "batch_size==1, len(dataset)=={} == len(dataloader)=={}".format(
            dataset_len,dataloader_len
        )
    )

    logger.info("start evaluating:")
    logger.info("use config: {}".format(cfg_path))
    logger.info("use device: {}".format(device))
    convertor = EvalFmtCvtor("vidvrd",score_merge="mul",segment_eval=True)
    relation_results = dict()
    for batch_data in tqdm(dataloader):
        batch_data = tuple(to_device_func(data_list) for data_list in batch_data)
        # for simplicity, we set batch_size = 1 at inference time
        (
            segment_tag,
            traj_pair_info,
            rel_pos_feat,
            traj_embds,   # (n_det_aft_filter,256)
            det_traj_info,
            labels
        ) = tuple(bd[0] for bd in batch_data)
        tids_wrt_oritraj,tids_wrt_trajembds,trajpair_unionembds = traj_pair_info
        traj_features = det_traj_info["features"]
        traj_bboxes = det_traj_info["bboxes"]
        traj_starts = det_traj_info["fstarts"]
        with torch.no_grad():
            traj_cls_ids,traj_scores = model_traj.forward_inference_bsz1(traj_features)  # (n_det,) # before filter
            pred_cls_ids,pred_scores = model_pred.forward_inference(traj_pair_info,rel_pos_feat,traj_embds) # (n_pair_aft_filter,)
            s_ids_ori = tids_wrt_oritraj[:,0]
            o_ids_ori = tids_wrt_oritraj[:,1]
            
            s_ids = tids_wrt_trajembds[:,0]
            o_ids = tids_wrt_trajembds[:,1]
            s_cls_ids = traj_cls_ids[s_ids]
            o_cls_ids = traj_cls_ids[o_ids]
            s_scores = traj_scores[s_ids]
            o_scores = traj_scores[o_ids]

            pr_5tuple = torch.stack(
                [pred_cls_ids,s_cls_ids,o_cls_ids,s_ids_ori,o_ids_ori],dim=-1
            )  # shape == (n_pair,5)
            pr_scores = torch.stack(
                [pred_scores,s_scores,o_scores],dim=-1
            ) # shpae == (n_pair,3)
        result_per_seg = convertor.to_eval_json_format(
            segment_tag,
            pr_5tuple.cpu(),
            pr_scores.cpu(),
            [tb.cpu() for tb in traj_bboxes],
            traj_starts.cpu(),
            preserve_debug_info=False
        )
        relation_results.update(result_per_seg)
    
    _eval_relation_detection_openvoc(
        logger=logger,
        prediction_results=relation_results,
        segment_eval=True,
        eval_split_traj=eval_split_traj,
        eval_split_pred=eval_split_pred,
        # traj_upbd=True,
        # pred_upbd=True
    )

    if save_json_results:
        save_path = os.path.join(output_dir,f"VidVRDtest_segments_relation_results_{save_tag}.json")
        logger.info("save results to {}".format(save_path))
        logger.info("saving ...")
        with open(save_path,'w') as f:
            json.dump(relation_results,f)
        logger.info("results saved.")
    
    logger.info(f"log saved at {log_path}")
    logger.handlers.clear()



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
    

    gt_relations = load_json("datasets/GT_json_for_eval/VidVRDtest_gts.json")
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,prediction_results,viou_threshold=0.5)
    # logger.info(f"mAP:{mean_ap}, Retection Recall:{rec_at_n}, Tagging Precision: {mprec_at_n}")
    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path_traj", type=str,help="...")
    parser.add_argument("--ckpt_path_pred", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--cuda", type=int,help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--json_results_path", type=str,help="...")
    args = parser.parse_args()


    if args.json_results_path is not None:
        eval_relation(
            json_results_path=args.json_results_path,
        )
    else:
        inference_then_eval(
            args.cfg_path,
            args.ckpt_path_traj,
            args.ckpt_path_pred,
            args.output_dir,
            args.cuda,
            args.eval_split_traj,
            args.eval_split_pred,
            args.save_tag,
            args.save_json_results
        )

    '''
    python tools/eval_relation_cls.py \
        --json_results_path /home/gkf/project/VideoGraph/training_dir_reorganized/vidvrd/model_0v10_pku_i3dclsme2_cachePKUv2/VidORval_predict_relations_topk10_pTrue_epoch70.json
    
    python tools/eval_relation_cls.py \
        --segment_eval \
        --eval_split_pred base \
        --json_results_path experiments/OpenVocRelCls/ALPro_unionbox_embds/VidVRDtest_segments_relation_results_TrajAllPredBase.json

    python tools/eval_relation_cls.py \
        --cfg_path  experiments/OpenVocRelCls_reAssignLabel/cfg_vidvrd_relcls.py\
        --ckpt_path_traj experiments/ALPro_teacher/model_OpenVoc_w15BS128_epoch_50.pth \
        --ckpt_path_pred experiments/OpenVocRelCls_reAssignLabel/model_vpoi_th0.7_bs128_epoch_50.pth \
        --output_dir experiments/OpenVocRelCls/OpenVocRelCls_reAssignLabel \
        --eval_split_pred base \
        --cuda 1 \
        --save_tag TrajAllPredBase
    

    python tools/eval_relation_cls.py \
        --cfg_path  experiments/OpenVocRelCls/cfg_vidvrd_relcls.py\
        --ckpt_path_traj experiments/ALPro_teacher/model_OpenVoc_w15BS128_epoch_50.pth \
        --ckpt_path_pred experiments/OpenVocRelCls/ablation_vpoi0.5/model_vpoi0.5_bs128_epoch_50.pth \
        --output_dir experiments/OpenVocRelCls/ablation_vpoi0.5 \
        --eval_split_pred novel \
        --cuda 3 \
        --save_tag TrajAllPredNovel


    export TMPDIR=/tmp/$USER 
    tensorboard --logdir=/home/gkf/project/VidVRD-OpenVoc/experiments/OpenVocRelCls/ALPro_unionbox_embds/logfile --port=6006 --bind_all
    
    /home/gkf/project/VidVRD-OpenVoc/experiments/OpenVocRelCls/ablation_train_cfg_1/logfile
    
tensorboard --logdir_spec=epoch50:/home/gkf/project/VidVRD-OpenVoc/experiments/OpenVocRelCls/ALPro_unionbox_embds/logfile,\
epoch80:/home/gkf/project/VidVRD-OpenVoc/experiments/OpenVocRelCls/ablation_train_cfg_1/logfile \
--port=6007 --bind_all

    '''