# Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs

Official implementation (based on Pytorch) of CVPR2022 paper [Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs](https://arxiv.org/abs/2112.04222).


# Datasets

# Evaluation:

# Training


# TODO
paper : 
the code is still being organized (an initial version will be completed before March 28, 2022).

# TODO 
- add code for training
- add explanation for some term, e.g., "proposal" "use_pku"
- change the term slots to bins
- Explain the EntiNameEmb and classeme and avg_clsme
- explain the format of TrajProposal's feature, e.g., traj_classeme = traj_features[:,:,self.dim_feat:]
- clean up utils_func
- All scores are truncated to 4 decimal places (not rounded)

# Data to release
- I3D feature of VidOR train & val around 6G
- VidOR traj `.np` files (OnlyPos) (this has been released, around 12G)
- VidVRD traj `.np` files (with feature) around 20G
- cache file for train & val (for vidor)
    - v9 for val (around 15G)
    - v7clsme for train (14 parts, around 130G in total)
- do not release cache file for vidvrd (they can generate them using VidVRD traj `.np` files)

