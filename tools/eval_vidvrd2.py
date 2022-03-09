import root_path
import argparse
import os 
import json
from tqdm import tqdm

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

    '''