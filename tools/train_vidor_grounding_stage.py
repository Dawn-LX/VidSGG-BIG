
import pickle

import root_path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter       
from tqdm import tqdm
from collections import defaultdict

from VORG_DataParallel import DataParallel as VORG_DataParallel
from configs.config_parser import parse_config_py
from importlib import import_module
from dataloaders.dataloader_vidor_v3 import Dataset
from utils.utils_func import create_logger

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
    DEBUG = import_module(model_class_path).DEBUG

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


    dataset = Dataset(**dataset_config)
    ###
    if all_cfgs["extra_config"]["dataloader_name"] == "dataloader_vidor_v3":
        assert dataset.cache_tag == "v9"
    else:
        assert dataset.cache_tag == "v7_with_clsme"
    ###

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



if __name__ == "__main__":
    cfg_path = "training_dir_reorganized/vidor/groundOnly_0/config_.py"
    model_class_path = "Tempformer_model/model_groundOnly_0.py"
    # ckpt_path = "training_dir_reorganized/vidor/model_0v4/model_epoch_40.pth"
    train(model_class_path,cfg_path,device_ids=[0],from_checkpoint=False,ckpt_path=None)