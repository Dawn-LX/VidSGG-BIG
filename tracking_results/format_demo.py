import numpy as np
file_path = "deepSORT/tracking_results/VidORtrain_freq1_m60s0.3_part01/0000_2401075277.npy"
track_results = np.load(file_path,allow_pickle=True)
# each `.npy` file contains all the bounding boxes in a video, the bounding box position is recorded as xywh (top-left x,y, and width & height)
# track_results.shape == (7310,) 7310 is the number of total bbox in this video
# each item of track_results is a list (len==6 or len==12). 
# if all of these list have a length of 12, then it has the shape (7310,12)

for box_info in track_results:
    assert len(box_info) == 6 or len(box_info) == 12

    frame_id = box_info[0]
    tid = box_info[1]  # tracklet id, Not necessarily continuous, e.g., a video may have 3 tracklets which have ids of 1,3,4
    tracklet_xywh = box_info[2:6] # tracklet xywh predicted by deepSORT algorithm
        
    if len(box_info) == 12:
        confidence = box_info[6]  # confidence score
        category_id = box_info[7] # object category id obtained from the detector (MEGA)
        xywh = box_info[8:12]     # object bbox from the detector (MEGA)

        # appearance_feature = box_info[12:]
        # the appearance feature (i.e., RoI pooled feature) from the detector
        # NOTE  However, we do not release these features due their large capacity 
        # and you can extract these features by the provided weight of detector.
        
