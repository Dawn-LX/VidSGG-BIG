import root_path

from tqdm import tqdm
import argparse
import torch

from dataloaders.dataloader_vidor_v3 import Dataset as Dataset_vidor
from dataloaders.dataloader_vidvrd import Dataset as Dataset_vidvrd
from dataloaders.dataloader_vidvrd import Dataset_pku,Dataset_pku_i3d

from utils.utils_func import parse_config_py

def demo(dataset_class,dataset_config):
    ## construct dataset
    dataset = dataset_class(**dataset_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        collate_fn=dataset.collator_func,
        shuffle=False,
        num_workers=2
    )


    for proposal_list,gt_graph_list in tqdm(dataloader):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--split", type=str,help="...")
    parser.add_argument("--dataset_class", type=str,help="...")
    args = parser.parse_args()

    all_cfgs = parse_config_py(args.cfg_path)

    dataset_config = all_cfgs[f"{args.dataset_class}_{args.split}_dataset_config"]

    if args.dataset_class == "vidvrd":
        dataset_class = Dataset_vidvrd
    elif args.dataset_class == "pku":
        dataset_class = Dataset_pku
    elif args.dataset_class == "pku_i3d":
        dataset_class = Dataset_pku_i3d
    elif args.dataset_class == "vidor":
        dataset_class = Dataset_vidor
    else:
        assert False

    demo(dataset_class,dataset_config)

    '''
    python tools/dataloader_demo.py \
        --cfg_path experiments/demo/config_.py \
        --split test \
        --dataset_class pku_i3d
    '''