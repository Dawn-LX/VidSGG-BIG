# Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs

Pytorch implementation of our paper [Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs](https://arxiv.org/abs/2112.04222), which is accepted by CVPR2022.

We also won the 1st place of Video Relation Understanding (VRU) Grand Challenge in ACM Multimedia 2021, with a simplified version of our model.(The code for object tracklets generation is available at [here](https://github.com/Dawn-LX/VidVRD-tracklets))

# Requirements
Python == 3.7 or later, Pytorch == 1.6 or later, for other basic packages, just run the project and download whatever needed.

# Datasets
Download the [ImageNet-VidVRD](https://xdshang.github.io/docs/imagenet-vidvrd.html) dataset and [VidOR](https://xdshang.github.io/docs/vidor.html) dataset, and put them in the following folder as

```
├── dataloaders
│   ├── dataloader_vidvrd.py
│   └── ...
├── datasets
│   ├── cache                       # cache file for our dataloaders
│   ├── vidvrd-dataset
│   │   ├── train
│   │   ├── test
│   │   └── videos
│   ├── vidor-dataset
│   │   ├── annotation
│   │   └── videos
│   └── GT_json_for_eval
│       ├── VidORval_gts.json       # GT josn for evlauate
│       └── VidVRDtest_gts.json
├── experiments   
├── models
├── ...

```

# Verify tracklets data & feature preparation by running dataloader_demo
This section helps you download the tracklets data and place them correctly, as well as set the dataloader's config correctly. Successfully run the `tools/dataloader_demo.py` to verify all data & configs are set correctly.

**NOTE** we use the term `proposal` in our code to represent tracklet proposals in video-level, which is totally different with the concept of "proposal" in "proposal-based methods" in our paper. In our paper, we use "proposals to represent paired subject-object tracklet segments. In contrast, here the term `proposal` in our code represents long-term object tracklets in video-level (i.e., without sliding window or video segments).

## Tracklet data for VidVRD 
1. Download the tracklet with features at here:  [train (to be released)](), [test](https://pan.zju.edu.cn/share/694f908a22fff11c037eb50876)

2. Download the tracklet with features used in ["Beyond Short-Term Snippet: Video Relation Detection with Spatio-Temporal Global Context"](https://pkumyd.github.io/paper/CVPR2020_VideoVRD.pdf) at the author's personal page [here](http://www.muyadong.com/publication.html). (**NOTE** we use the term `pku` (i.e., Peking University) in our code to refer to their tracklets & features)

3. The tracklet with features are in `VidVRD_test_every1frames` (ours), `VidVRD_train_every1frames` (ours), `preprocess_data/tracking/videovrd_detect_tracking` (PKU, both train & test), in whcih each `.npy` file corresponds to a video and contains all the tracklets in that video. The I3D features of tracklets are in `preprocess_data/tracking/videovrd_i3d` (PKU, both train & test).
Put them under the dir of this project (or any other position if you use absolute path).

4. modify the config file at `experiments/demo/config_.py`, where `proposal_dir` is the dir of tracklet with features, `i3d_dir` is the dir of tracklets' I3D features, and `ann_dir` is `datasets/vidvrd-dataset`.

5. Verify all data & configs are set correctly. e.g., for PKU's tracklets with I3D features, run the following commands: (refer to `tools/dataloader_demo.py` for more details.):
    ```
    python tools/dataloader_demo.py \
            --cfg_path experiments/demo/config_.py \
            --split test \
            --dataset_class pku_i3d
    ```

## Tracklet data for VidOR 

- TODO

# Evaluation:

1. first generate the GT json file for evaluation:
    
    for vidvrd:
    ```
    python VidVRD-helper/prepare_gts_for_eval.py \
        --dataset_type vidvrd \
        --save_path datasets/GT_json_for_eval/VidVRDtest_gts.json
    ```
    for vidor:
    ```
    python VidVRD-helper/prepare_gts_for_eval.py \
        --dataset_type vidor \
        --save_path datasets/GT_json_for_eval/VidORval_gts.json
    ```
2. Download model weights for different exps [here](https://drive.google.com/file/d/1vE-cQrNUrpSKrWC94orbbpVLkvuDKFwm/view?usp=sharing), and put them in the `experiments/` dir.
3. Refer to `experiments/readme.md` for the correspondence between the exp ids and the table ids in our paper.
4. For **VidVRD**, run the following commands to evaluate different exps: (refer to `tools/eval_vidvrd.py` for more details)

    e.g., for exp1
    ```
    python tools/eval_vidvrd.py \
        --cfg_path experiments/exp1/config_.py \
        --ckpt_path experiments/exp1/model_epoch_80.pth \
        --use_pku \
        --cuda 1 \
        --save_tag debug
    ```
5. For **VidOR**, refer to `tools/eval_vidor.py` for more details. 

    Run the following commands to evaluate BIG-C (i.e., only the classification stage):
    ```
    python tools/eval_vidor.py \
        --eval_cls_only \
        --cfg_path experiments/exp4/config_.py \
        --ckpt_path experiments/exp4/model_epoch_60.pth \
        --save_tag epoch60_debug \
        --cuda 1
    ```
    Run the following commands to evaluate BIG based on the output of cls stage (you need run BIG-C first and save the `infer_results`).
    ```
    python tools/eval_vidor.py \
        --cfg_path experiments/grounding_weights/config_.py \
        --ckpt_path experiments/grounding_weights/model_epoch_70.pth \
        --output_dir experiments/exp4_with_grounding \
        --cls_stage_result_path experiments/exp4/VidORval_infer_results_topk3_epoch60_debug.pkl \
        --save_tag with_grd_epoch70 \
        --cuda 1
    ```
    Run the following commands to evaluate the fraction recall (refer to table-6 in our paper, you need run BIG first and save the `hit_infos`).
    ```
    python tools/eval_fraction_recall.py \
        --cfg_path experiments/grounding_weights/config_.py \
        --hit_info_path  experiments/exp5_with_grounding/VidORval_hit_infos_aft_grd_with_grd_epoch70.pkl
    ```

**NOTE** 
- We also provide another evaluation scripts (i.e., `tools/eval_vidvrd_our_gt.py` and `tools/eval_vidor_our_gt.py`). The main difference lies in the process of constructing GT tracklets (i.e., from frame-level bbox annotations to video-level tracklets GTs). Compared to VidVRD-helper's GTs, here we perform linear interpolation for fragmented GT tracklets. Consequently the evaluation results have slight differences.
- Nevertheless, the results reported in our paper are evaluated with VidVRD-helper's GTs (i.e., `tools/eval_vidvrd.py` and `tools/eval_vidor.py`) to ensure fair comparisons.
    
# Training (TODO)

the code for training is still being organized (an initial version will be completed before March 28, 2022).

## Data to release
- I3D feature of VidOR train & val around 6G
- VidOR traj `.npy` files (OnlyPos) (this has been released, around 12G)
- VidVRD traj `.npy` files (with feature) around 20G
- cache file for train & val (for vidor)
    - v9 for val (around 15G)
    - v7clsme for train (14 parts, around 130G in total)
- do not release cache file for vidvrd (they can generate them using VidVRD traj `.npy` files)


# Citation
If our work is helpful for your research, please cite our publication:
```
@inproceedings{gao2021classification,
  title={Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs},
  author={Gao, Kaifeng and Chen, Long and Niu, Yulei and Shao, Jian and Xiao, Jun},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2022}
}
```

# TODO 
- add code for training
- add explanation for some term, e.g., "proposal" "use_pku"
- change the term slots to bins
- Explain the EntiNameEmb and classeme and avg_clsme
- explain the format of TrajProposal's feature, e.g., traj_classeme = traj_features[:,:,self.dim_feat:]
- clean up utils_func
- All scores are truncated to 4 decimal places (not rounded)


