# Object tracklets for VidOR train/validation/test set

- you can download the object tracklets for VidOR dataset [here](https://drive.google.com/drive/folders/1wWkzHlhYcZPQR4fUMTTJEn2SVVnhGFch?usp=sharing)
- the object bounding boxes and categories are obtained by MEGA
- the tracking algorithm we used is deepSORT
- these tracklets (these `.npy` files) only contain tracklet postions and object categories.
- the appearance features (e.g., RoI pooled feature) for each bbox are not released due their large capacity.
- please refer to ``format_demo.py`` for detailed format.

## for VidOR train set
VidORtrain_freq1_m60s0.3_part01 ~ VidORtrain_freq1_m60s0.3_part14

each part contains 500 videos (500 .npy files)

## for VidOR validation set

VidORval_freq1_m60s0.3, which contains 835 videos

## for VidOR test set

VidORtest_freq1_m60s0.3, which contains 2165 videos

## Explanation of file name and parameters

we explain some parameters in the file names (e.g., VidORtrain_freq1_m60s0.3_part01/0000_2401075277.npy)

- freq1: the sample rate is 1, i.e., we run MEGA and deepSORT on **each frame** of the video (despite of the large redundancy )
- m60: the parameter `max_age` in deepSORT tracker, which controls ``maximum number of missed misses before a track is deleted''. m60 means that we allow 60 missed frames for a tracklet

- s0.3: the score threshold is deepSORT, the bounding box with confidence lower than 0.3 will be deleted and will not be considered  for tracking.

- 0000_2401075277: 0000 is the group id and 2401075277 is the video id, i.e., this "0000_2401075277.npy" corresponds to "0000/2401075277.mp4" in VidOR train set.


# Object tracklets for VidVRD train/test set
 - VidVRD_train_every1frames: 800 `.npy` files
 - VidVRD_test_every1frames: 200 `.npy` files
 - these `.npy` files have the same format as that in VidOR. Refer to `format_demo.py`
 - The parameters we used in deepSORT for VidVRD: m30s0.3