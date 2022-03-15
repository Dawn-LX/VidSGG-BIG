
######### for PKU without I3D ################
pku_train_dataset_config = dict(
    split = "train",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_detect_tracking",
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v1"
)


pku_test_dataset_config = dict(
    split = "test",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_detect_tracking",
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v1"
)

######### for PKU without I3D ################

pku_i3d_train_dataset_config = dict(
    split = "train",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_detect_tracking",
    i3d_dir = "proposals/vidvrd-dataset/PKU_beyond/videovrd_i3d",
    dim_i3d = 832,
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v2"
)


pku_i3d_test_dataset_config = dict(
    split = "test",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_detect_tracking",
    i3d_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/PKU_beyond/videovrd_i3d",
    dim_i3d = 832,
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v2"
)

######### for MEGA ################
vidvrd_train_dataset_config = dict(
    split = "train",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/miss30_minscore0p3/VidVRD_train_every1frames",
    dim_boxfeature = 1024,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "MEGA"
)


vidvrd_test_dataset_config = dict(
    split = "test",
    ann_dir = "/home/gkf/project/VideoGraph/datasets/vidvrd-dataset",
    proposal_dir = "/home/gkf/project/VideoGraph/proposals/vidvrd-dataset/miss30_minscore0p3/VidVRD_test_every1frames",
    dim_boxfeature = 1024,
    min_frames_th = 5,
    max_proposal = 150,
    max_preds = 100,
    cache_tag = "MEGA"
)


if __name__ == "__main__":
    pass
