# Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs

Pytorch implementation of our paper [Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs](https://arxiv.org/abs/2112.04222), which is accepted by CVPR2022.

We also won the 1st place of Video Relation Understanding (VRU) Grand Challenge in ACM Multimedia 2021, with a simplified version of our model.(The code for object tracklets generation is available at [here](https://github.com/Dawn-LX/VidVRD-tracklets))

# Datasets

# Evaluation:

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


# TODO 
- add code for training
- add explanation for some term, e.g., "proposal" "use_pku"
- change the term slots to bins
- Explain the EntiNameEmb and classeme and avg_clsme
- explain the format of TrajProposal's feature, e.g., traj_classeme = traj_features[:,:,self.dim_feat:]
- clean up utils_func
- All scores are truncated to 4 decimal places (not rounded)



