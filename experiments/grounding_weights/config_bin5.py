
model_config = dict(
    dim_feat = 1024,
    dim_clsme = 300,
    dim_hidden = 128,
    num_bins  = 5,
    EntiNameEmb_path = "prepared_data/vidor_EntiNameEmb.npy",
    PredNameEmb_path = "prepared_data/vidor_PredNameEmb.npy",
    loss_factor = dict(
        classification = 1.0,
        centerness = 1.0,
        regression = 1.0,
    )
)


test_dataset_config = dict(
    split = "val",
    video_dir = '/home/gkf/project/VidVRD_VidOR/vidor-dataset/val_videos',
    ann_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation", 
    proposal_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1", 
    classeme_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORval_freq1_classeme",
    video_feature_dir = "/home/gkf/project/I3D_Pytorch/I3D_clip_features/clip16_overlap0.5_val",
    max_proposal = 180,
    max_preds = 200,
    score_th = 0.4,
    dim_boxfeature = 1024,
    min_frames_th = 15,
    cache_tag = "MEGAv9_m60s0.3_freq1"
)

train_dataset_config = dict(
    split = "train",
    ann_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation",
    video_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/train_videos",
    classeme_dir = "/home/gkf/project/deepSORT/tracking_results/miss60_minscore0p3/VidORtrain_freq1_classeme",
    video_feature_dir = "/home/gkf/project/I3D_Pytorch/I3D_clip_features/clip16_overlap0.5",
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
    batch_size          = 8,
    total_epoch         = 80,
    initial_lr          = 5e-5,
    lr_decay            = 0.2,
    epoch_lr_milestones = [40,60],
)


inference_config = dict(
    score_th = 0.9, # for temporal_pooling
    tiou_th = 0.5,  # for temporal_pooling
    bins_th = 0.2,# for multi-bin NMS
    nms_th = 0.8,  # for multi-bin NMS
    eval_tiouths = 0.5  # for evaluate the grounding module itself only.
)

if __name__ == "__main__":
    print(model_config)
    print(train_dataset_config)
