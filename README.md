# TODO
the code is still being organized

# TODO 
- add code for training
- add explanation for some term, e.g., "proposal" "use_pku"
- change the term slots to bins
- Explain the EntiNameEmb and classeme and avg_clsme
- explain the format of TrajProposal's feature, e.g., traj_classeme = traj_features[:,:,self.dim_feat:]
- clean up utils_func

# Data to release
- I3D feature of VidOR train & val around 6G
- VidOR traj `.np` files (OnlyPos) (this has been released, around 12G)
- VidVRD traj `.np` files (with feature) around 20G
- cache file for train & val (for vidor)
    - v9 for val (around 15G)
    - v7clsme for train (14 parts, around 130G in total)
- do not release cache file for vidvrd (they can generate them using VidVRD traj `.np` files)


# 考虑一下 v7_with_clsme 的val-set

我们应该release的是 v7_with_clsme train-set cache, 但是我们evaluate的是 v9 的 val-set
最好的做法是统一到一个dataloader里，然后 train-set cache 和 val-set cache 都保持原来paper里用的 (i.e., v7_clsme for train and v9 for val) (做完grounding 再做这个)

