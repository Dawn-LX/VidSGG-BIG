# TODO
the code is still being organized

# TODO 
- add code for training
- add explanation for some term, e.g., "proposal" "use_pku"
- Explain the EntiNameEmb and classeme and avg_clsme
- explain the format of TrajProposal's feature, e.g., traj_classeme = traj_features[:,:,self.dim_feat:]

# 考虑一下 v7_with_clsme 的val-set

我们应该release的是 v7_with_clsme train-set cache, 但是我们evaluate的是 v9 的 val-set
最好的做法是统一到一个dataloader里，然后 train-set cache 和 val-set cache 都保持原来paper里用的 (i.e., v7_clsme for train and v9 for val)
