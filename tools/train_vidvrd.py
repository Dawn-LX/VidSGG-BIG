
import root_path
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter       
from tqdm import tqdm
from collections import defaultdict

from dataloaders.dataloader_vidvrd import Dataset,Dataset_pku,Dataset_pku_i3d
from models import BIG_C_vidvrd

from utils.DataParallel import VidSGG_DataParallel
from utils.utils_func import create_logger,parse_config_py

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


def train(
    cfg_path,
    use_pku =True,
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
    model = BIG_C_vidvrd(model_config,is_train=True)
    # logger.info(model)
    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("number of anchors = {}".format(model.num_anchors))
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--use_pku",action="store_true",default=False,help="...")
    parser.add_argument("--from_checkpoint",action="store_true",default=False,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")    
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    
    args = parser.parse_args()

    train(
        args.cfg_path,
        use_pku=args.use_pku,
        experiment_dir=args.output_dir,
        save_tag=args.save_tag
    )
    
    '''
    ## for exp1
    CUDA_VISIBLE_DEVICES=0 python tools/train_vidvrd.py \
        --cfg_path experiments/exp1/config_.py \
        --use_pku \
        --save_tag retrain
    
    ## for exp2
    CUDA_VISIBLE_DEVICES=1,2 python tools/train_vidvrd.py \
        --cfg_path experiments/exp2/config_.py \
        --use_pku \
        --save_tag retrain
    
    ## for exp3
    CUDA_VISIBLE_DEVICES=2,3 python tools/train_vidvrd.py \
        --cfg_path experiments/exp3/config_.py \
        --save_tag retrain
    
    '''
