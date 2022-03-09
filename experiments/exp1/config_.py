
model_config = dict(
    num_enti_cats   = 36,
    num_pred_cats   = 133,
    dim_ffn         = 512,
    dim_enti        = 512,
    dim_pred        = 512,
    dim_att         = 512,
    dim_feat        = 2048,         # dimension of each bbox's RoI feature, depend on the detector
    dim_clsme       = 300,
    enco_pool_len   = 4,
    n_enco_layers   = 2,
    n_deco_layers   = 6,
    n_att_head      = 8,
    num_querys      = 192,
    neg_weight      = 0.1,
    positive_vIoU_th= 0.5,
    EntiNameEmb_path= "prepared_data/vidvrd_EntiNameEmb_pku.npy",
    bias_matrix_path= "prepared_data/pred_bias_matrix_vidvrd_pku.npy",
    cost_coeff_dict = dict(
        classification      = 1.0,
        adj_matrix          = 30.0,
    ),
    loss_coeff_dict = dict(         # loss coefficient dictionary        
        classification      = 1.0,
        adj_matrix          = 30.0,
    )
)
train_dataset_config = dict(
    split = "train",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_detect_tracking",
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v1"
)


test_dataset_config = dict(
    split = "test",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_detect_tracking",
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v1"
)

## train-dataset cache_tag: v7_with_clsme

train_config = dict(
    batch_size          = 8,
    total_epoch         = 100,
    initial_lr          = 1e-4,
    lr_decay            = 0.2,
    epoch_lr_milestones = [80],
)

inference_config = dict(
    topk = 10,
)

if __name__ == "__main__":
    pass
