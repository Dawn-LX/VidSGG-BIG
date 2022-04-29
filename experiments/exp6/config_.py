
model_config = dict(
    dataset_type    = "VidOR",
    num_enti_cats   = 81,
    num_pred_cats   = 51,
    dim_ffn         = 512,
    dim_enti        = 512,
    dim_pred        = 512,
    dim_att         = 512,
    dim_feat        = 1024,         # dimension of each bbox's RoI feature, depend on the detector
    dim_clsme       = 300,
    enco_pool_len   = 4,
    positive_vIoU_th= 0.5,
    rt_triplets_topk = -1,  # -1 for return all
    EntiNameEmb_path= None,
    use_clsme       = True,
    bias_matrix_path= "prepared_data/pred_bias_matrix_vidor.npy",
)

test_dataset_config = dict(
    split = "val",
    video_dir = '/home/gkf/project/VidVRD_VidOR/vidor-dataset/val_videos',
    ann_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation", 
    proposal_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1", 
    classeme_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1_classeme",
    max_proposal = 180,
    max_preds = 200,
    score_th = 0.4,
    dim_boxfeature = 1024,
    min_frames_th = 15,
    cache_tag = "MEGAv9_m60s0.3_freq1"
)
# test-dataset_cache_tag: v9

train_dataset_config = dict(
    split = "train",
    ann_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation",
    video_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/train_videos",
    classeme_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORtrain_freq1_classeme",
    proposal_dir = {
        0:"proposals/miss60_minscore0p3/VidORtrain_freq1_part01",
        1:"proposals/miss60_minscore0p3/VidORtrain_freq1_part02",
        2:"proposals/miss60_minscore0p3/VidORtrain_freq1_part03",
        3:"proposals/miss60_minscore0p3/VidORtrain_freq1_part04",
        4:"proposals/miss60_minscore0p3/VidORtrain_freq1_part05",
        5:"proposals/miss60_minscore0p3/VidORtrain_freq1_part06",
        6:"proposals/miss60_minscore0p3/VidORtrain_freq1_part07",
        7:"proposals/miss60_minscore0p3/VidORtrain_freq1_part08",
        8:"proposals/miss60_minscore0p3/VidORtrain_freq1_part09",
        9:"proposals/miss60_minscore0p3/VidORtrain_freq1_part10",
        10:"proposals/miss60_minscore0p3/VidORtrain_freq1_part11",
        11:"proposals/miss60_minscore0p3/VidORtrain_freq1_part12",
        12:"proposals/miss60_minscore0p3/VidORtrain_freq1_part13",
        13:"proposals/miss60_minscore0p3/VidORtrain_freq1_part14",
    },
    cache_dir = "datasets/cache",
    cache_tag = "MEGAv7",
    dim_boxfeature = 1024,
    min_frames_th = 15,
    max_proposal = 180,
    max_preds = 200,
    score_th = 0.4   
)

train_config = dict(
    batch_size          = 4,
    total_epoch         = 80,
    initial_lr          = 5e-5,
    lr_decay            = 0.2,
    epoch_lr_milestones = [50],
)

inference_config = dict(
    topk = 3,
)
extra_config = dict(
    dataloader_name = "dataloader_vidor"
)
if __name__ == "__main__":
    print(model_config)
    print(train_dataset_config)
