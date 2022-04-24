
# this is a draft code
# the training code is still being organized
# 

import root_path

import pickle
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter       
from tqdm import tqdm
from collections import defaultdict

from VORG_DataParallel import DataParallel as VORG_DataParallel
from configs.config_parser import parse_config_py
from importlib import import_module
from dataloaders.dataloader_vidvrd_pkuv2 import Dataset
# from dataloaders.dataloader_vidvrd_pku import Dataset
from utils.evaluate import EvalFmtCvtor
from evaluation import eval_visual_relation,eval_visual_relation_v2
from utils.utils_func import create_logger,collator_func_sort

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

 
def train(model_class_path,cfg_path,experiment_dir=None, device_ids = [0],from_checkpoint = False,ckpt_path = None):
    ## import model class 
    temp = model_class_path.split('.')[0].split('/')
    model_class_path = ".".join(temp)
    TempFormer = import_module(model_class_path).TempFormer
    # TempFormer = import_module(model_class_path).VORG

    ## create dirs and logger
    if experiment_dir == None:
        experiment_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(experiment_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not from_checkpoint:
        os.system("rm {}events*".format(log_dir))
    
    writer = SummaryWriter(log_dir)
    log_path = os.path.join(log_dir,'train.log')
    logger = create_logger(log_path)

    ## load configs
    all_cfgs = parse_config_py(cfg_path)
    model_config = all_cfgs["model_config"]
    dataset_config = all_cfgs["train_dataset_config"]
    train_config = all_cfgs["train_config"]

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(train_config)

    ## construct model
    model = TempFormer(model_config,is_train=True)
    logger.info(model)
    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("number of anchors = {}".format(model.num_anchors))
    logger.info("number of model.parameters: total:{},trainable:{}".format(total_num,trainable_num))

    model = VORG_DataParallel(model,device_ids=device_ids)
    model = model.cuda("cuda:{}".format(device_ids[0]))

    # training configs

    batch_size          = train_config["batch_size"]
    total_epoch         = train_config["total_epoch"]
    initial_lr          = train_config["initial_lr"]
    lr_decay            = train_config["lr_decay"]
    epoch_lr_milestones = train_config["epoch_lr_milestones"]

    logger.info(all_cfgs["extra_config"])
    dataset = Dataset(**dataset_config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=dataset.collator_func,
        shuffle=True,
        num_workers=2
    )

    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    logger.info(
        "len(dataset)=={},batch_size=={},len(dataloader)=={},{}x{}={}".format(
            dataset_len,batch_size,dataloader_len,batch_size,dataloader_len,batch_size*dataloader_len
        )
    )
    

    milestones = [int(m*dataset_len/batch_size) for m in epoch_lr_milestones]
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=lr_decay)


    if from_checkpoint:
        model,optimizer,scheduler,crt_epoch,batch_size_ = load_checkpoint(model,optimizer,scheduler,ckpt_path)
        # assert batch_size == batch_size_ , "batch_size from checkpoint not match : {} != {}"
        if batch_size != batch_size_:
            logger.warning(
                "!!!Warning!!! batch_size from checkpoint not match : {} != {}".format(batch_size,batch_size_)
            )
        logger.info("checkpoint load from {}".format(ckpt_path))
    else:
        crt_epoch = 0

    logger.info("start training:")
    logger.info("cfg_path = {}".format(cfg_path))
    logger.info("weights will be saved in experiment_dir = {}".format(experiment_dir))


    it=0
    for epoch in tqdm(range(total_epoch)):
        if epoch < crt_epoch:
            it+=dataloader_len
            continue

        epoch_loss = defaultdict(list)
        for proposal_list,gt_graph_list in dataloader:
            
            optimizer.zero_grad()
            combined_loss, each_loss_term = model(proposal_list,gt_graph_list)
            # average results from muti-gpus
            combined_loss = combined_loss.mean()
            each_loss_term = {k:v.mean() for k,v in each_loss_term.items()}
            
            loss_str = "epoch={},iter={},loss={:.4f}; ".format(epoch,it,combined_loss.item())
            writer.add_scalar('Iter/total_loss', combined_loss.item(), it)
            epoch_loss["total_loss"].append(combined_loss.item())
            combined_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            scheduler.step()

            for k,v in each_loss_term.items():
                epoch_loss[k].append(v.item())
                loss_str += "{}:{:.4f}; ".format(k,v.item())
                writer.add_scalar('Iter/{}'.format(k), v.item(), it)
            loss_str += "lr={}".format(optimizer.param_groups[0]["lr"])
            if it % 10 == 0:
                logger.info(loss_str)
            it+=1
    
    
        epoch_loss_str = "mean_loss_epoch={}: ".format(epoch)
        for k,v in epoch_loss.items():  
            v = np.mean(v)
            writer.add_scalar('Epoch/{}'.format(k), v, epoch)
            epoch_loss_str += "{}:{:.4f}; ".format(k,v)
        logger.info(epoch_loss_str)
        
        if epoch >0 and epoch % 10 == 0:
            save_path = os.path.join(experiment_dir,'model_epoch_{}.pth'.format(epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            logger.info("checkpoint is saved: {}".format(save_path))
    
    save_path = os.path.join(experiment_dir,'model_epoch_{}.pth'.format(total_epoch))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    logger.info("checkpoint is saved: {}".format(save_path))



def eval(model_class_path,cfg_path,weight_path,epoch_tag="",use_pku=False,
    experiment_dir=None,gpu_id = 0,save_infer_result=False,save_relation_json=False):
    print(use_pku)
    ## import model class 
    temp = model_class_path.split('.')[0].split('/')
    model_class_path = ".".join(temp)
    TempFormer = import_module(model_class_path).TempFormer
    # TempFormer = import_module(model_class_path).VORG


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
    use_post=infer_config["use_post"]
    topk=infer_config["topk"]

    # create logger
    # save_tag = "topk-{}_regr-{}_postpro-{}_epoch-{}".format(topk,use_regr,use_post,epoch) # this filename is too long
    save_tag = "topk{}_p{}_epoch{}".format(topk,use_post,epoch_tag)

    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(infer_config)


    model = TempFormer(model_config,is_train=False)
    # logger.info(model)

    model = model.cuda(device)
    state_dict = torch.load(weight_path,map_location=torch.device('cpu'))
    state_dict = state_dict["model_state_dict"]
    state_dict_ = {}
    for k in state_dict.keys():
        state_dict_[k[7:]] = state_dict[k]
    # state_dict_ = replace_state_dict_keys(state_dict_)
    model.load_state_dict(state_dict_)
    model.eval()


    # import numpy as np
    # EntiNameEmb_path = "/home/gkf/VidSGG/tools/vidvrd_EntiNameEmb.npy"
    # EntiNameEmb = np.load(EntiNameEmb_path)
    # EntiNameEmb = torch.from_numpy(EntiNameEmb).float()
    
    # EntiNameEmb_model = model.EntiNameEmb.cpu()
    # mask = EntiNameEmb == EntiNameEmb_model
    # print(EntiNameEmb)
    # print(EntiNameEmb_model)
    # print(torch.all(mask))

    # assert False

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
            batch_triplets = model(proposal_list,use_post=use_post,topk=topk)

        assert len(proposal_list) == 1
        video_name = proposal_list[0].video_name
        if batch_triplets[0] is None:
            infer_result.update({video_name:None})
            continue

        (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            # uniq_pred_confs,    # shape == (n_unique,)
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
        save_path = os.path.join(experiment_dir,'VidVRDval_infer_result_{}.pkl'.format(save_tag))
        with open(save_path,'wb') as f:
            pickle.dump(infer_result_for_save,f)
        logger.info("infer_result saved at {}".format(save_path))

    logger.info("start convert format for evaluate...")


    convertor_all = EvalFmtCvtor("vidvrd",eval_separately=False)
    
    gt_relations = {}
    predict_relations = {}
    for proposal_list,gt_graph_list in tqdm(dataloader):
        assert len(proposal_list) == 1
        proposal = proposal_list[0]
        gt_graph = gt_graph_list[0]

        pr_triplet = infer_result[proposal.video_name]
        
        
        # for eval overall

        pr_result = convertor_all.to_eval_format_pr(proposal,pr_triplet,use_pku=use_pku)
        predict_relations.update(pr_result) 
        gt_result = convertor_all.to_eval_format_gt(gt_graph)
        gt_relations.update(gt_result["overall"])
    

    logger.info('For part:**overall**, Computing average precision AP over {} videos...'.format(len(gt_relations)))
    mean_ap, rec_at_n, mprec_at_n,hit_infos = eval_visual_relation_v2(gt_relations,predict_relations,viou_threshold=0.5,is_print=False,det_nreturns=[50,100])

    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))

    save_path = os.path.join(experiment_dir,'VidORval_hit_infos_{}.pkl'.format(save_tag))
    with open(save_path,'wb') as f:
        pickle.dump(hit_infos,f)
    logger.info("hit_infos have been saved at {}".format(save_path))
    
    logger.info("evaluate results and log file have been saved at {}".format(log_path))

    
    if save_relation_json:
        save_path = os.path.join(experiment_dir,'VidORval_predict_relations_{}.json'.format(save_tag))
        logger.info("saving predict_relations into {}...".format(save_path))
        with open(save_path,'w') as f:
            json.dump(predict_relations,f)
        logger.info("predict_relations have been saved at {}".format(save_path))
        

def eval_old(model_class_path,cfg_path,weight_path,epoch_tag="",
    experiment_dir=None,gpu_id = 0,save_infer_result=False,save_relation_result=False):
    
    ## import model class 
    temp = model_class_path.split('.')[0].split('/')
    model_class_path = ".".join(temp)
    # TempFormer = import_module(model_class_path).TempFormer
    TempFormer = import_module(model_class_path).VORG


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
    use_post=infer_config["use_post"]
    topk=infer_config["topk"]

    # create logger
    # save_tag = "topk-{}_regr-{}_postpro-{}_epoch-{}".format(topk,use_regr,use_post,epoch) # this filename is too long
    save_tag = "topk{}_p{}_epoch{}".format(topk,use_post,epoch_tag)

    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(infer_config)


    model = TempFormer(model_config,is_train=False)
    logger.info(model)

    model = model.cuda(device)
    state_dict = torch.load(weight_path,map_location=torch.device('cpu'))
    state_dict = state_dict["model_state_dict"]
    state_dict_ = {}
    for k in state_dict.keys():
        state_dict_[k[7:]] = state_dict[k]
    model.load_state_dict(state_dict_)
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


    predict_relations = dict()
    gt_relations = dict()
    ratio_list  =[]
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal_list = [p.to(device) for p in proposal_list]
        # gt_graph_list = [g.to(cuda0) for g in gt_graph_list]

        predict_per_batch = model(proposal_list,use_post=use_post)
        result,ratio = predict_per_batch
        ratio_list += ratio
        predict_relations.update(result)
        # for pro in proposal_list:
        #     print(pro.video_name,pro.video_len)
        
        for gt in gt_graph_list:
            if gt.num_trajs==0 or gt.num_preds==0:
                gt_relations.update({gt.video_name:[]})
            else:
                gt_res = gt.to_eval_format()
                assert gt_res.keys() == result.keys()
                gt_relations.update(gt_res)

    mean_ap, rec_at_n, mprec_at_n,hit_infos = eval_visual_relation_v2(gt_relations,predict_relations,viou_threshold=0.5,is_print=False,det_nreturns=[50,100])

    logger.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    logger.info('detection recall: {}'.format(rec_at_n))
    logger.info('tagging precision: {}'.format(mprec_at_n))

    save_path = os.path.join(experiment_dir,'VidORval_hit_infos_{}.pkl'.format(save_tag))
    with open(save_path,'wb') as f:
        pickle.dump(hit_infos,f)
    logger.info("hit_infos have been saved at {}".format(save_path))
    
    logger.info("evaluate results and log file have been saved at {}".format(log_path))

    
    if save_relation_result:
        save_path = os.path.join(experiment_dir,'VidORval_predict_relations_{}.pkl'.format(save_tag))
        logger.info("saving predict_relations into {}...".format(save_path))
        with open(save_path,'wb') as f:
            pickle.dump(predict_relations,f)
        logger.info("predict_relations have been saved at {}".format(save_path))
  


def test_import(model_class_path,cfg_path):
    all_cfgs = parse_config_py(cfg_path)
    model_config = all_cfgs["model_config"]
    dataset_config = all_cfgs["train_dataset_config"]
    train_config = all_cfgs["train_config"]

    temp = model_class_path.split('.')[0].split('/')
    model_class_path = ".".join(temp)

    TempFormer = import_module(model_class_path).TempFormer
    print(TempFormer)

    model = TempFormer(model_config,is_train=True)
    print(model)


def test_lr(model_class_path,cfg_path):
    
    all_cfgs = parse_config_py(cfg_path)
    model_config = all_cfgs["model_config"]
    dataset_config = all_cfgs["train_dataset_config"]
    train_config = all_cfgs["train_config"]

    temp = model_class_path.split('.')[0].split('/')
    model_class_path = ".".join(temp)

    TempFormer = import_module(model_class_path).TempFormer
    model = TempFormer(model_config,is_train=True)
    model = VORG_DataParallel(model,device_ids=[1,2])

    fc_regr_params = [x[1] for x in model.named_parameters() if "bbox_head" in x[0]]
    main_params = [x[1] for x in model.named_parameters() if not("bbox_head" in x[0])]


    optim = torch.optim.Adam(
        [
            {"params": main_params},
            {"params": fc_regr_params, "lr": 1e-5},
            
        ],
        lr=5e-5,
    )

    print(optim.param_groups,len(optim.param_groups))


def replace_state_dict_keys(state_dict):
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

    ##### ablation for PKU RoI+I3D
    # cfg_path = "training_dir_reorganized/vidvrd/model_0v10_pku_i3d_cachePKUv2/config_.py"
    # model_class_path = "Tempformer_model/model_0v10_pku_i3d.py"
    # # model_class_path = "Tempformer_model/model_vrd02.py"
    # train(model_class_path,cfg_path,device_ids=[1,2],from_checkpoint=False,ckpt_path=None)

    # cfg_path = "training_dir_reorganized/vidvrd/model_0v10_cachePKUv1_rightcatid/config_.py"
    # model_class_path = "Tempformer_model/model_0v10.py"
    # weight_path = "training_dir_reorganized/vidvrd/model_0v10_cachePKUv1_rightcatid/model_epoch_80.pth"
    # train(model_class_path,cfg_path,device_ids=[0],from_checkpoint=False,ckpt_path=None)

    ### ablation for PKU RoI+I3D+Lan
    cfg_path = "training_dir_reorganized/vidvrd/model_0v10_pku_i3dclsme2_cachePKUv2/config_.py"
    model_class_path = "Tempformer_model/model_0v10_pku_i3dclsme2.py"
    weight_path = "training_dir_reorganized/vidvrd/model_0v10_pku_i3dclsme2_cachePKUv2/model_epoch_70.pth"
    # train(model_class_path,cfg_path,device_ids=[1,2],from_checkpoint=False,ckpt_path=None)


    
    # # # # # weight_path = "training_dir_reorganized/vidvrd/v16_svrd_ced_att_MEGA_cache5/paper1933_weight/model_epoch_80.pth"
    epoch_tag = weight_path.split('.')[0].split('_')[-1] + "debug"
    eval(model_class_path,cfg_path,weight_path,epoch_tag,gpu_id=1,use_pku=True,save_relation_json=False)
