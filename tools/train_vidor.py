# this is a draft code
# the training code is still being organized

import root_path
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter       
from tqdm import tqdm
from collections import defaultdict

from dataloaders.dataloader_vidor import Dataset
from models import BIG_C_vidor,Base_C,DEBUG

from utils.DataParallel import VidSGG_DataParallel
from utils.utils_func import create_logger,parse_config_py,dura_intersection_ts,vIoU_ts


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





def prepare_data(gt_graph):
    if gt_graph.num_trajs==0 or gt_graph.num_preds==0:
        return None,None
    
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

    target = pred_durations/video_len

    return gt_5tuples,target



def trajid2pairid(num_prop):
    mask = torch.ones(size=(num_prop,num_prop),dtype=torch.bool)
    mask[range(num_prop),range(num_prop)] = 0
    pair_ids = mask.nonzero(as_tuple=False)

    return pair_ids

def prop_pair_to_gt_pred(dataset,positive_vIoU_th):

    assert len(dataset.proposal_dir) == 14, "we suggest that assign labels for all 14 parts together"
    dataloader0 = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        collate_fn = dataset.collator_func,
        shuffle=False
    )

    num_total_hit_gt_traj = 0
    num_total_gt_traj = 0
    
    num_total_hit_gt_pred = 0
    num_total_gt_pred = 0
    

    label_maps = {}
    for proposal_list,gt_graph_list in tqdm(dataloader0):
        gt_graph = gt_graph_list[0]
        proposal = proposal_list[0]
        video_name = gt_graph.video_name
        if gt_graph.num_trajs==0 or gt_graph.num_preds==0:
            label_maps[video_name] = None
            continue

        pr_trajbboxes,pr_trajduras = proposal.bboxes_list,proposal.traj_durations
        gt_trajbboxes,gt_trajduras = gt_graph.traj_bboxes,gt_graph.traj_durations
        num_gt_enti = len(gt_trajbboxes)
        inter_dura,dura_mask = dura_intersection_ts(pr_trajduras,gt_trajduras)  # shape == (n_traj,n_gt_traj,2)
        
        inter_dura_p = inter_dura - pr_trajduras[:,0,None,None]  # convert to relative duration
        inter_dura_g = inter_dura - gt_trajduras[None,:,0,None]
        
        pids,gids = dura_mask.nonzero(as_tuple=True)  # row, col : pid,gid
        viou_matrix = torch.zeros_like(dura_mask,dtype=torch.float)   # (n_prop, n_gt_traj)
        for pid,gid in zip(pids.tolist(),gids.tolist()):
            dura_p = inter_dura_p[pid,gid,:]
            dura_g = inter_dura_g[pid,gid,:]
            bboxes_p = pr_trajbboxes[pid]
            bboxes_g = gt_trajbboxes[gid]
            viou_matrix[pid,gid] = vIoU_ts(bboxes_p,bboxes_g,dura_p,dura_g)
        


        num_prop = proposal.num_proposals
        n_gt_traj = gt_graph.num_trajs
        assert viou_matrix.shape == (num_prop,n_gt_traj)
        hit_mask = (viou_matrix > positive_vIoU_th)
        n_hit_gt_traj = torch.any(hit_mask,dim=0).sum()  # (n_gt_traj,) --> scalar

        num_total_gt_traj += n_gt_traj
        num_total_hit_gt_traj += n_hit_gt_traj

        pair_ids = trajid2pairid(num_prop)
        
        gt_5tuples,_ = prepare_data(gt_graph)

        n_gt_pred = gt_5tuples.shape[0]
        assert gt_graph.num_preds == n_gt_pred
        

        trajids2gt5tuples = defaultdict(list)
        n_hit_gt_pred = 0
        for gt_idx in range(n_gt_pred):
            gt_sid,gt_oid = gt_5tuples[gt_idx,3:].tolist()
            # print(gt_5tuples[gt_idx,:])
            # print(gt_5tuples[gt_idx,3:],gt_5tuples[gt_idx,-2:])
            # break
            is_hit = False
            for pair_id in pair_ids:
                sid,oid = pair_id.tolist()
                viou_s = viou_matrix[sid,gt_sid]
                viou_o = viou_matrix[oid,gt_oid]
                if (viou_s > positive_vIoU_th) and (viou_o > positive_vIoU_th):
                    trajids2gt5tuples[(sid,oid)].append(gt_5tuples[gt_idx].cpu())
                    is_hit = True
            if is_hit:
                n_hit_gt_pred += 1
        num_total_hit_gt_pred += n_hit_gt_pred
        num_total_gt_pred += n_gt_pred

        label_maps[video_name] = trajids2gt5tuples

    print("traj",num_total_hit_gt_traj,num_total_gt_traj,num_total_hit_gt_traj/num_total_gt_traj)
    print("pred",num_total_hit_gt_pred,num_total_gt_pred,num_total_hit_gt_pred/num_total_gt_pred)
    
    del dataloader0

    return label_maps



 
def train_baseline(
    cfg_path,
    experiment_dir=None,
    save_tag="",
    from_checkpoint = False,
    ckpt_path = None
):
    

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
    model = Base_C(model_config,is_train=True)
    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("number of model.parameters: total:{},trainable:{}".format(total_num,trainable_num))

    model = model.cuda()
    device=torch.device("cuda")
    # device = torch.device("cpu")


    # training configs
    batch_size          = train_config["batch_size"]
    total_epoch         = train_config["total_epoch"]
    initial_lr          = train_config["initial_lr"]
    lr_decay            = train_config["lr_decay"]
    epoch_lr_milestones = train_config["epoch_lr_milestones"]

    dataset = Dataset(**dataset_config)

    
    num_pred_cats = model_config["num_pred_cats"]  # include background
    positive_vIoU_th = model_config["positive_vIoU_th"]
    label_map_path = "datasets/cache/VidORtrain_label_maps_vIoU{:.2f}.pkl".format(positive_vIoU_th)

    if os.path.exists(label_map_path):
        logger.info("load assigned labels... from cache file : {}".format(label_map_path))
        with open(label_map_path,'rb') as f:
            label_map = pickle.load(f)
    else:
        logger.info("no cache file found, start assigning labels... save_path = {}".format(label_map_path))
        label_map = prop_pair_to_gt_pred(dataset,positive_vIoU_th)
        with open(label_map_path,'wb') as f:
            pickle.dump(label_map,f)
    
    trajpair2labels = {}
    for video_name, trajids2gt5tuples in label_map.items():
        num_pairs = len(trajids2gt5tuples.keys())
        if num_pairs == 0:
            trajpair2labels[video_name] = None
            continue
        pairid2trajids = torch.zeros(size=(num_pairs,2),dtype=torch.long)
        multihot = torch.zeros(size=(num_pairs,num_pred_cats))
        for idx,(k,v) in enumerate(trajids2gt5tuples.items()):
            pairid2trajids[idx,:] = torch.tensor(k)
            gt5tuples = torch.stack(v,dim=0)  # (num_gt,5)
            pred_catids = gt5tuples[:,0]  # (num_gt,)
            multihot[idx,pred_catids] = 1   # TODO add negative samples
        
        trajpair2labels[video_name] = (pairid2trajids,multihot)

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

    logger.info("start training (using 1 GPU, for Base-C only single gpu training is supported now):")
    logger.info("cfg_path = {}".format(cfg_path))
    logger.info("weights will be saved in experiment_dir = {}".format(experiment_dir))


    it=0
    for epoch in tqdm(range(total_epoch)):
        if epoch < crt_epoch:
            it+=dataloader_len
            continue

        epoch_loss = defaultdict(list)
        for proposal_list,_ in dataloader:
            video_names = [p.video_name for p in proposal_list]
            video_names = [name for name in video_names if trajpair2labels[name] is not None]
            if len(video_names) == 0:
                continue

            pos_id_list = [trajpair2labels[name][0].to(device) for name in video_names]
            label_list = [trajpair2labels[name][1].to(device) for name in video_names]
            proposal_list = [p.to(device) for p in proposal_list if trajpair2labels[p.video_name] is not None]

            
            optimizer.zero_grad()
            combined_loss, each_loss_term = model(proposal_list,pos_id_list,label_list)
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
            save_path = os.path.join(experiment_dir,'model_epoch_{}_{}.pth'.format(epoch,save_tag))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            logger.info("checkpoint is saved: {}".format(save_path))
    
    save_path = os.path.join(experiment_dir,'model_epoch_{}_{}.pth'.format(total_epoch,save_tag))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    logger.info("checkpoint is saved: {}".format(save_path))



def train_cls_stage(
    cfg_path,
    experiment_dir=None,
    save_tag = "",
    from_checkpoint = False,
    ckpt_path = None
):

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
    model = BIG_C_vidor(model_config,is_train=True)
    # logger.info(model)
    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("number of num_querys = {}".format(model.num_querys))
    logger.info("number of model.parameters: total:{},trainable:{}".format(total_num,trainable_num))

    device_ids = list(range(torch.cuda.device_count()))
    model = VidSGG_DataParallel(model,device_ids=device_ids)
    model = model.cuda()

    # training configs
    batch_size          = train_config["batch_size"]
    total_epoch         = train_config["total_epoch"]
    initial_lr          = train_config["initial_lr"]
    lr_decay            = train_config["lr_decay"]
    epoch_lr_milestones = train_config["epoch_lr_milestones"]


    dataset = Dataset(**dataset_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=dataset.collator_func,
        shuffle=True,
        num_workers=4
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
            save_path = os.path.join(experiment_dir,'model_epoch_{}_{}.pth'.format(epoch,save_tag))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            logger.info("checkpoint is saved: {}".format(save_path))
    
    save_path = os.path.join(experiment_dir,'model_epoch_{}_{}.pth'.format(total_epoch,save_tag))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    logger.info("checkpoint is saved: {}".format(save_path))
    logger.info(f"log saved at {log_path}")
    logger.handlers.clear()


 
def train_grounding_stage(
    cfg_path,
    experiment_dir=None, 
    save_tag = "",
    from_checkpoint = False,
    ckpt_path = None
):

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
    model = DEBUG(model_config,is_train=True)
    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("number of model.parameters: total:{},trainable:{}".format(total_num,trainable_num))

    device_ids = list(range(torch.cuda.device_count()))
    model = VidSGG_DataParallel(model,device_ids=device_ids)
    model = model.cuda()

    # training configs

    batch_size          = train_config["batch_size"]
    total_epoch         = train_config["total_epoch"]
    initial_lr          = train_config["initial_lr"]
    lr_decay            = train_config["lr_decay"]
    epoch_lr_milestones = train_config["epoch_lr_milestones"]


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
        for video_feature_list,proposal_list,gt_graph_list in dataloader:
            
            optimizer.zero_grad()
            combined_loss, each_loss_term = model(video_feature_list,gt_graph_list)
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
            save_path = os.path.join(experiment_dir,'model_epoch_{}_{}.pth'.format(epoch,save_tag))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            logger.info("checkpoint is saved: {}".format(save_path))
    
    save_path = os.path.join(experiment_dir,'model_epoch_{}_{}.pth'.format(total_epoch,save_tag))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    logger.info("checkpoint is saved: {}".format(save_path))
    logger.info(f"log saved at {log_path}")
    logger.handlers.clear()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--from_checkpoint",action="store_true",default=False,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")    
    parser.add_argument("--train_baseline", action="store_true",default=False,help="...")
    parser.add_argument("--train_grounding", action="store_true",default=False,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    
    args = parser.parse_args()

    if args.train_baseline:
        train_baseline(
            args.cfg_path,
            experiment_dir=args.output_dir,
            save_tag=args.save_tag,
            from_checkpoint=args.from_checkpoint,
            ckpt_path=args.ckpt_path
        )
        exit(0)
    
    if args.train_grounding:
        train_grounding_stage(
            args.cfg_path,
            experiment_dir=args.output_dir,
            save_tag=args.save_tag,
            from_checkpoint=args.from_checkpoint,
            ckpt_path=args.ckpt_path
        )
        exit(0)

    train_cls_stage(
        args.cfg_path,
        experiment_dir=args.output_dir,
        save_tag=args.save_tag,
        from_checkpoint=args.from_checkpoint,
        ckpt_path=args.ckpt_path
    )

    '''
    ## for exp4 (80 epochs, around 24 hours for 2 RTX 2080Ti with batch_size=4) or maybe one RTX 2080 Ti ?, I forgotten this ... 
    CUDA_VISIBLE_DEVICES=1,2 python tools/train_vidor.py \
        --cfg_path experiments/exp4/config_.py \
        --save_tag retrain
    
    ## for exp5 (similar time as exp4)
    CUDA_VISIBLE_DEVICES=1,2 python tools/train_vidor.py \
        --cfg_path experiments/exp5/config_.py \
        --save_tag retrain
    
    ## for exp6 (80 epochs, around 6.5 hours for 1 RTX 2080Ti with batch_size=4)
    CUDA_VISIBLE_DEVICES=1 python tools/train_vidor.py \
        --train_baseline \
        --cfg_path experiments/exp6/config_.py \
        --save_tag retrain
    

    ## for train grounding stage (80 epochs, around 11 hours for 2 RTX 2080Ti with batch_size=8)
    CUDA_VISIBLE_DEVICES=2,3 python tools/train_vidor.py \
        --train_grounding \
        --cfg_path experiments/grounding_weights/config_.py \
        --save_tag retrain
    '''
