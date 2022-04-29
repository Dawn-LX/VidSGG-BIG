
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import pickle
from copy import deepcopy
import random
from collections import defaultdict
from utils.utils_func import is_overlap, merge_duration_list,linear_interpolation
from utils.categories_v2 import vidvrd_CatName2Id,vidvrd_PredName2Id
    
class TrajProposal(object):
    def __init__(self,video_name,
        cat_ids,traj_bboxes_with_score,traj_durations,roi_features,MAX_PROPOSAL):
        """
        roi_features： list[np.arry], len==num_proposals, shape == (num_frames,dim)
        """
        self.MAX_PROPOSAL = MAX_PROPOSAL
        self.video_name = video_name
        assert len(cat_ids) == len(traj_bboxes_with_score)
        assert len(cat_ids) == len(roi_features)
        assert len(cat_ids) == len(traj_durations)
        if len(cat_ids) == 0:
            self.num_proposals = 0
            print("video:{} has no proposal".format(video_name))
            return
        
        self.cat_ids = torch.LongTensor(cat_ids)                                  # len == num_proposals
        self.scores,self.bboxes_list = self._preprocess_boxes(traj_bboxes_with_score)    # len == num_proposals
        # list[float], list[tensor], traj_bboxes.shape == (num_frames,4)
        self.traj_durations = torch.tensor(traj_durations,dtype=torch.long)  # --> shape == (num_proposals, 2)
        self.traj_durations[:,1] -= 1  # convert to closed interval
        self.features_list = [torch.FloatTensor(r) for r in roi_features]  # len==num_proposals, shape == (num_frames,dim_feat) traj_durations

        
        # score_clipping:
        self.scores = torch.FloatTensor(self.scores)
        index = torch.argsort(self.scores,descending=True)
        index = index[:self.MAX_PROPOSAL]
        self.scores = self.scores[index]
        self.bboxes_list = [self.bboxes_list[ii] for ii in index]
        self.traj_durations = self.traj_durations[index]
        self.features_list = [self.features_list[ii] for ii in index]
        self.cat_ids = self.cat_ids[index]
        
        self.num_proposals = len(self.bboxes_list)
        if self.num_proposals > self.MAX_PROPOSAL:
            self.num_proposals = self.MAX_PROPOSAL
        
        self.dim_feat = self.features_list[0].shape[1] if self.num_proposals > 0 else "[]"

    def _preprocess_boxes(self,traj_bboxes_with_score):
        scores = []
        traj_bboxes = []
        for proposal in traj_bboxes_with_score:
            # proposal: list[[xmin,ymin,xmax,ymax,score]]  len == num_frames
            bboxes = torch.FloatTensor([p[0:4] for p in proposal])
            traj_bboxes.append(bboxes)

            scores.append(
                sum([p[4] for p in proposal])/len(proposal)
            )
        return scores,traj_bboxes
    def to(self,device):
        if self.num_proposals == 0:
            return self
        
        self.cat_ids = self.cat_ids.to(device)
        self.scores = self.scores.to(device)
        self.traj_durations = self.traj_durations.to(device)
        for i in range(self.num_proposals):
            self.bboxes_list[i] = self.bboxes_list[i].to(device)
            self.features_list[i] = self.features_list[i].to(device)
        
        return self

    
    def __repr__(self):

        return "TrajProposal[{},num_proposals={},feature_dim={}]".format(self.video_name,self.num_proposals,self.dim_feat)

class VideoGraph(object):
    def __init__(self,video_info,split,
        traj_cat_ids,traj_durations,traj_bboxes,
        pred_cat_ids,pred_durations,
        adj_matrix_subject,adj_matrix_object,MAX_PREDS):
        self.video_name,self.video_len,self.video_wh = video_info
        self.MAX_PREDS = MAX_PREDS

        assert len(traj_cat_ids) == len(traj_durations)
        assert len(traj_cat_ids) == len(traj_bboxes)
        assert len(pred_cat_ids) == len(pred_durations)
        self.num_trajs = len(traj_cat_ids)
        self.num_preds = len(pred_cat_ids)

        self.traj_cat_ids = torch.LongTensor(traj_cat_ids)     # shape == (num_trajs,)
        self.traj_durations = torch.tensor(traj_durations)     # shape == (num_trajs,2), 
        self.traj_durations[:,1] -= 1                          # convert to closed interval
        self.traj_bboxes = [torch.FloatTensor(b) for b in traj_bboxes] # list[tensor]  len == num_trajs, shape == (num_frames,4)
        self.pred_cat_ids = torch.LongTensor(pred_cat_ids)     # shape == (num_preds,)
        self.pred_durations = torch.tensor(pred_durations).float()     # shape == (num_preds,2),  
        self.pred_durations[:,1] -= 1                          # convert to closed interval
        self.n_frames_list = [traj.shape[0] for traj in self.traj_bboxes]
        self.max_frames = max(self.n_frames_list)

        assert adj_matrix_object.shape == adj_matrix_subject.shape
        adj_s = torch.from_numpy(adj_matrix_subject)  # shape == (num_preds,num_trajs)
        adj_o = torch.from_numpy(adj_matrix_object)
        self.adj_matrix = torch.stack([adj_s,adj_o],dim=0).float()   # shape = (2,num_preds,num_trajs)

        if split == "train" and (self.num_preds > self.MAX_PREDS):
            self.pred_cat_ids = self.pred_cat_ids[:self.MAX_PREDS]
            self.pred_durations = self.pred_durations[:self.MAX_PREDS,:]
            self.adj_matrix = self.adj_matrix[:,:self.MAX_PREDS,:]
            self.num_preds = self.pred_cat_ids.shape[0]
        
        
    def _prepare_pred_duras(self,pred_durations):
        start = []
        end = []
        for dura in pred_durations:
            start.append(dura[0])
            end.append(dura[1])
        
        start = torch.FloatTensor(start)
        end = torch.FloatTensor(end)
        gt_duras = torch.stack([start,end],dim=1)   # shape == (num_gt_pred,2)
        return gt_duras

    def to(self,device):
        self.traj_cat_ids = self.traj_cat_ids.to(device)
        self.pred_cat_ids = self.pred_cat_ids.to(device)
        self.traj_durations = self.traj_durations.to(device)
        self.pred_durations = self.pred_durations.to(device)        # shape == (num_preds,2),  start_fid, end_fid
        self.adj_matrix = self.adj_matrix.to(device)
        for i in range(self.num_trajs):
            self.traj_bboxes[i] = self.traj_bboxes[i].to(device)
        return self

    def __repr__(self):
        return "VideoGraph[num_trajs={},num_preds={}]".format(self.num_trajs,self.num_preds)


class Dataset(object):

    def __init__(self,split,ann_dir,proposal_dir,
        dim_boxfeature,min_frames_th,max_proposal,max_preds,cache_tag
    ):
        self.split = self._get_split(split)  # self.split in ["train","test"]
        self.proposal_dir = proposal_dir  # e.g., "proposals/vidvrd-dataset/miss30_minscore0p3/VidVRD_test_every1frames"
        self.dim_boxfeature = dim_boxfeature
        self.min_frames_th = min_frames_th
        self.max_proposal = max_proposal
        self.max_preds = max_preds
        self.cache_tag = cache_tag
        self.cache_dir = "datasets/cache"
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)


        if self.split == "train":
            self.video_ann_dir = os.path.join(ann_dir,'train')  # e.g., "datasets/vidvrd-dataset/train/"
        else: # self.split == "test":
            self.video_ann_dir = os.path.join(ann_dir,'test')    # e.g., "datasets/vidvrd-dataset/test/"

        self.video_name_list = self._prepare_video_names()

        cache_path = self.cache_tag + "_" + "VidVRD{}_th_{}-{}-{}".format(self.split,self.min_frames_th,self.max_proposal,self.max_preds)
        cache_path = os.path.join(self.cache_dir,cache_path + ".pkl")
        if os.path.exists(cache_path):
            print("load all data into memory, from cache file {}".format(cache_path))
            with open(cache_path,'rb') as f:
                data_dict = pickle.load(f)
        else:
            print("no cache file find, preparing data... {}".format(cache_path))
            data_dict = dict()
            for video_name in tqdm(self.video_name_list):
                data_dict[video_name] = self.get_data(video_name)
            with open(cache_path,'wb') as f:
                pickle.dump(data_dict,f)
            print("all data have been saved as cache file: {}".format(cache_path))
        
        self.data_dict = data_dict
        print("all data have been loaded into memory")   
    
    def _get_split(self,split):
        train = {x:"train" for x in ["train","training"]}
        test = {x:"test" for x in ["test","testing"]}
        split_dict = {}
        for x in [train,test]:
            split_dict.update(x)
        
        try:
            s = split_dict[split.lower()]
        except:
            assert False, "only support split in {}".format(split_dict.keys())
        
        return s

    def _prepare_video_names(self):
        video_name_list = os.listdir(self.video_ann_dir)
        video_name_list = sorted([v.split('.')[0] for v in video_name_list])
        return video_name_list

    def __getitem__(self,idx):
        video_name = self.video_name_list[idx]
        if (video_name == "ILSVRC2015_train_00082000") or (video_name == "ILSVRC2015_train_00790000"):  # videos in trainset, which cost too much gpu memory
            idx = random.randint(0,len(self.video_name_list)-1)
            return self.__getitem__(idx)


        traj_proposal, gt_graph = deepcopy(self.data_dict[video_name]) 


        if traj_proposal.num_proposals > 0 or self.split == "test":
            return traj_proposal, gt_graph
        else:
            idx = random.randint(0,len(self.video_name_list)-1)
            # print("video:{} has no proposal, random choose {} instead".format(video_name,self.video_name_list[idx]))
            return self.__getitem__(idx)

    
    def __len__(self):
        return len(self.video_name_list)
    
    def get_data(self,video_name):
        traj_proposal = self._get_proposal(video_name)
        gt_graph = self._get_gt_graph(video_name)
        
        traj_proposal.video_len = gt_graph.video_len    # TODO add traj_proposal.video_len separately and assert `traj_proposal.video_len == gt_graph.video_len`
        traj_proposal.video_wh  = gt_graph.video_wh
        return traj_proposal, gt_graph

    def _get_proposal(self,video_name):
        track_res_path = os.path.join(self.proposal_dir,video_name+".npy") #ILSVRC2015_train_00010001.npy
        track_res = np.load(track_res_path,allow_pickle=True)
        trajs = {box_info[1]:{} for box_info in track_res}
        for tid in trajs.keys():  
            trajs[tid]["frame_ids"] = []
            trajs[tid]["bboxes"] = []
            trajs[tid]["roi_features"] = []
            trajs[tid]["category_id"] = []   # 如果某个tid只有len==6的box_info，那就无法获取 category_id ，默认为背景

        for box_info in track_res:
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 6 or len(box_info) == 12 + self.dim_boxfeature,"len(box_info)=={}".format(len(box_info))
            
            frame_id = box_info[0]
            tid = box_info[1]
            tracklet_xywh = box_info[2:6]
            xmin_t,ymin_t,w_t,h_t = tracklet_xywh
            xmax_t = xmin_t + w_t
            ymax_t = ymin_t + h_t
            bbox_t = [xmin_t,ymin_t,xmax_t,ymax_t]
            confidence = float(0)
            if len(box_info) == 12 + self.dim_boxfeature:
                confidence = box_info[6]
                cat_id = box_info[7]
                xywh = box_info[8:12]
                xmin,ymin,w,h = xywh
                xmax = xmin+w
                ymax = ymin+h
                bbox = [(xmin+xmin_t)/2, (ymin+ymin_t)/2, (xmax+xmax_t)/2,(ymax+ymax_t)/2]
                roi_feature = box_info[12:]
                trajs[tid]["category_id"].append(cat_id)
                trajs[tid]["roi_features"].append(roi_feature)
            
            if len(box_info) == 6:
                bbox_t.append(confidence)
                trajs[tid]["bboxes"].append(bbox_t)
                trajs[tid]["roi_features"].append([0]*self.dim_boxfeature)
            else:
                bbox.append(confidence)
                trajs[tid]["bboxes"].append(bbox)
            trajs[tid]["frame_ids"].append(frame_id)    # 
    

        for tid in trajs.keys():
            if trajs[tid]["category_id"] == []:
                trajs[tid]["category_id"] = 0
            else:
                # print(trajs[tid]["category_id"])
                temp = np.argmax(np.bincount(trajs[tid]["category_id"]))  # 求众数
                trajs[tid]["category_id"] = int(temp)
            
            frame_ids = trajs[tid]["frame_ids"]
            start = min(frame_ids)
            end = max(frame_ids) + 1
            dura_len = end - start
            duration = (start,end)  # 前闭后开区间
            trajs[tid]["roi_features"] = np.array(trajs[tid]["roi_features"])
            trajs[tid]["bboxes"] = np.array(trajs[tid]["bboxes"])

            # 将太短的视为背景，后续过滤掉
            if len(frame_ids) < self.min_frames_th:
                trajs[tid]["category_id"] = 0
            else:
                trajs[tid]["duration"] = (start,end)
            
            # 对于非背景的traj， 看是否需要插值
            if trajs[tid]["category_id"] !=0 and len(frame_ids) != dura_len:
                trajs[tid]["roi_features"] = linear_interpolation(trajs[tid]["roi_features"],frame_ids)
                trajs[tid]["bboxes"] = linear_interpolation(trajs[tid]["bboxes"],frame_ids)
            
            if trajs[tid]["category_id"] !=0:
                assert len(trajs[tid]["bboxes"]) == dura_len

        # trajs = {k:v for k,v in trajs.items() if v["category_id"]!=0}
        cat_ids = []
        traj_boxes = []
        roi_features_list = []
        traj_durations = []
        for tid in trajs.keys():
            if trajs[tid]["category_id"] != 0:
                dura_len = trajs[tid]["duration"][1] - trajs[tid]["duration"][0]
                assert len(trajs[tid]["bboxes"]) == dura_len
                cat_ids.append(trajs[tid]["category_id"])
                traj_boxes.append(trajs[tid]["bboxes"])
                roi_features_list.append(trajs[tid]["roi_features"])
                traj_durations.append(trajs[tid]["duration"])
        
        return TrajProposal(video_name,cat_ids,traj_boxes,traj_durations,roi_features_list,self.max_proposal)
    
    def _get_gt_graph(self,video_name):
        video_ann_path = os.path.join(self.video_ann_dir, video_name + ".json") 

        ## 1. construct trajectory annotations from frame-level bbox annos
        if os.path.exists(video_ann_path):   
            with open(video_ann_path,'r') as f:
                video_anno = json.load(f)
        else:
            print(video_name,"not find its anno")
            raise NotImplementedError
        
        video_len = len(video_anno["trajectories"])
        video_wh = (video_anno["width"],video_anno["height"])

        traj_categories = video_anno["subject/objects"]      # tid not necessary 0 ~ len(traj_categories)-1
        # tid2category_map = [traj["category"] for traj in traj_categories] #  This is WRONG!
        tid2category_map = {traj["tid"]:traj["category"] for traj in traj_categories} # this is CORRECT
        # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}
        trajs = {traj["tid"]:{} for traj in traj_categories}

        for tid in trajs.keys():
            trajs[tid]["all_bboxes"] = []
            trajs[tid]["frame_ids"] = []
            trajs[tid]["cat_name"] = tid2category_map[tid]

        for frame_id,frame_anno in enumerate(video_anno["trajectories"]):
            for bbox_anno in frame_anno:
                tid = bbox_anno["tid"]
                category_name = tid2category_map[tid]
                category_id = vidvrd_CatName2Id[category_name]

                bbox = bbox_anno["bbox"]
                bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                trajs[tid]["all_bboxes"].append(bbox)
                trajs[tid]["frame_ids"].append(frame_id)
                trajs[tid]["category_id"] = category_id
        
        ## 2.linear_interpolation
        for tid in trajs.keys():
            frame_ids = trajs[tid]["frame_ids"]
            start,end = min(frame_ids),max(frame_ids)+1   # 前开后闭区间
            all_bboxes = np.array(trajs[tid]["all_bboxes"]).astype(np.float)
            all_bboxes = linear_interpolation(all_bboxes,frame_ids)
            trajs[tid]["all_bboxes"] = all_bboxes   # np.ndarray, shape == (dura_len,4)  
            trajs[tid]["duration"] = (start,end)    # dura_len == end - start

        traj_cat_ids = []
        traj_durations = []
        traj_bbox_list = []
        tid2idx_map = {}
        for idx,tid in enumerate(trajs.keys()):
            traj_cat_ids.append(trajs[tid]["category_id"])
            traj_durations.append(trajs[tid]["duration"])
            traj_bbox_list.append(trajs[tid]["all_bboxes"])
            tid2idx_map[tid] = idx
        traj_cat_ids = np.array(traj_cat_ids)       # shape == (num_trajs,)
        traj_durations = np.array(traj_durations)   # shape == (num_trajs,2)
        num_trajs = len(traj_cat_ids)
        
        # 3. merge relations
        # in the train-set of vidvrd, some long relations is annotated as short segments 
        # we merge them to one whole relation
        # e.g., a original relation might have a duration_list = [(195, 225), (225, 240), (375, 405), (390, 420)]
        # we merge it to [(195, 240), (375, 420)]
        # NOTE: train 的需要merge， 在vidvrd的train-set中， 一个大于30的duration都没有，在test-set中， long-duration没有被按照30一段标注
        preds = video_anno["relation_instances"]
        pred_cat_ids = []
        pred_durations = []
        trituple_list = []
        trituple2durations_dict = defaultdict(list)
        for pred in preds:
            predicate = pred["predicate"]
            subject_tid = pred["subject_tid"]
            object_tid = pred["object_tid"]
            trituple = str(subject_tid) + "-" + predicate + "-" +  str(object_tid)
            
            begin_fid = pred["begin_fid"]
            end_fid = pred["end_fid"]
            trituple2durations_dict[trituple].append((begin_fid,end_fid))
        
        for trituple,durations in trituple2durations_dict.items():
            merged_durations = merge_duration_list(durations)    # e.g., [(30,60),(60,90),(120,150)] --> [(30,90),(120,150)]
            trituple2durations_dict[trituple] = merged_durations

            pred_name = trituple.split('-')[1]
            pred_catid = vidvrd_PredName2Id[pred_name]

            for duration in merged_durations:
                trituple_list.append(trituple)
                pred_cat_ids.append(pred_catid)
                pred_durations.append(duration)
        
        num_preds = len(pred_cat_ids)
        pred_cat_ids = np.array(pred_cat_ids)  # shape == (num_preds,)
        pred_durations = np.array(pred_durations) # shape == (num_preds,2)
            
        # 2.3. construct adjacency matrix 
        adj_matrix_subject = np.zeros((num_preds,num_trajs),dtype=np.int)
        adj_matrix_object = np.zeros((num_preds,num_trajs),dtype=np.int)
        for idx in range(num_preds):
            trituple = trituple_list[idx]
            pred_duration = pred_durations[idx]

            subj_tid = int(trituple.split('-')[0])
            obj_tid = int(trituple.split('-')[-1])
            subj_idx = tid2idx_map[subj_tid]
            obj_idx = tid2idx_map[obj_tid]
            
            subj_duration = traj_durations[subj_idx]
            if is_overlap(pred_duration,subj_duration):
                adj_matrix_subject[idx,subj_idx] = 1
            
            obj_duration = traj_durations[obj_idx]
            if is_overlap(pred_duration,obj_duration): # is_overlap 可以用于 1-d np.ndarray
                adj_matrix_object[idx,obj_idx] = 1

        for row in adj_matrix_subject:
            assert np.sum(row) == 1, "video:{} not correct".format(video_name)
        
        for row in adj_matrix_object:
            assert np.sum(row) == 1, "video:{} not correct".format(video_name)
        
        video_info = (video_name,video_len,video_wh)

        return VideoGraph(video_info,self.split,
                traj_cat_ids,traj_durations,traj_bbox_list,
                pred_cat_ids,pred_durations,
                adj_matrix_subject,adj_matrix_object,self.max_preds
                )


    @property
    def collator_func(self):
        """
            batch is a list ,len(batch) == batch_size
            batch[i] is a tuple, batch[i][0],batch[i][1] is an object of class TrajProposal, class VideoGraph, respectively
            This function should be passed to the torch.utils.data.DataLoader

        """
        # batch_size = len(batch)
        # as for gramma of this , refer to `test_gramma/regonized/collator_func.py`
        def collator_func_v2(batch):
            batch = sorted(batch,key=lambda x: x[1].max_frames)

            batch_proposal = [b[0] for b in batch]
            batch_gt_graph = [b[1] for b in batch]

            return batch_proposal,batch_gt_graph
        

        return collator_func_v2

class Dataset_pku(Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

        assert self.cache_tag[:3].lower() == "pku"

    def _get_proposal(self,video_name):
        if video_name == "ILSVRC2015_train_00884000":  # PKU missing this file, use ours instead
            video_name = "ILSVRC2015_train_00884000" + "_myFaster18"

        track_res_path = os.path.join(self.proposal_dir,video_name+".npy") #ILSVRC2015_train_00010001.npy
        track_res = np.load(track_res_path,allow_pickle=True)
        trajs = {box_info[1]:{} for box_info in track_res}
        for tid in trajs.keys():  
            trajs[tid]["frame_ids"] = []
            trajs[tid]["bboxes"] = []
            trajs[tid]["roi_features"] = []
            trajs[tid]["category_id"] = []   # 如果某个tid只有len==6的box_info，那就无法获取 category_id ，默认为背景

        for box_info in track_res:
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 12 + self.dim_boxfeature,"len(box_info)=={}".format(len(box_info))
            
            frame_id = int(box_info[0])
            tid = int(box_info[1])
            tracklet_xywh = box_info[2:6]
            xmin_t,ymin_t,w_t,h_t = tracklet_xywh
            xmax_t = xmin_t + w_t
            ymax_t = ymin_t + h_t
            bbox_t = [xmin_t,ymin_t,xmax_t,ymax_t]
            confidence = box_info[6]
            cat_id = int(box_info[7])
            if cat_id <= 0:
                confidence = 0.0
                bbox = bbox_t + [confidence]
                roi_feature = [0]*self.dim_boxfeature
            else:
                xywh = box_info[8:12]
                xmin,ymin,w,h = xywh
                xmax = xmin+w
                ymax = ymin+h
                bbox = [(xmin+xmin_t)/2, (ymin+ymin_t)/2, (xmax+xmax_t)/2,(ymax+ymax_t)/2,confidence]
                roi_feature = box_info[12:]
                trajs[tid]["category_id"].append(cat_id)

            
            trajs[tid]["bboxes"].append(bbox)
            trajs[tid]["roi_features"].append(roi_feature)
            trajs[tid]["frame_ids"].append(frame_id)    # 
    

        for tid in trajs.keys():
            if trajs[tid]["category_id"] == []:
                trajs[tid]["category_id"] = 0
            else:
                # print(video_name,trajs[tid]["category_id"])
                temp = np.argmax(np.bincount(trajs[tid]["category_id"]))  # 求众数
                trajs[tid]["category_id"] = int(temp)
            
            frame_ids = trajs[tid]["frame_ids"]
            start = min(frame_ids)
            end = max(frame_ids) + 1
            dura_len = end - start
            duration = (start,end)  # 前闭后开区间
            trajs[tid]["roi_features"] = np.array(trajs[tid]["roi_features"])
            trajs[tid]["bboxes"] = np.array(trajs[tid]["bboxes"])

            # 将太短的视为背景，后续过滤掉
            if len(frame_ids) < self.min_frames_th:
                trajs[tid]["category_id"] = 0
            else:
                trajs[tid]["duration"] = (start,end)
            
            # 对于非背景的traj， 看是否需要插值
            if trajs[tid]["category_id"] !=0 and len(frame_ids) != dura_len:
                trajs[tid]["roi_features"] = linear_interpolation(trajs[tid]["roi_features"],frame_ids)
                trajs[tid]["bboxes"] = linear_interpolation(trajs[tid]["bboxes"],frame_ids)
            
            if trajs[tid]["category_id"] !=0:
                assert len(trajs[tid]["bboxes"]) == dura_len

        # trajs = {k:v for k,v in trajs.items() if v["category_id"]!=0}
        cat_ids = []
        traj_boxes = []
        roi_features_list = []
        traj_durations = []
        for tid in trajs.keys():
            if trajs[tid]["category_id"] != 0:
                dura_len = trajs[tid]["duration"][1] - trajs[tid]["duration"][0]
                assert len(trajs[tid]["bboxes"]) == dura_len
                cat_ids.append(trajs[tid]["category_id"])
                traj_boxes.append(trajs[tid]["bboxes"])
                roi_features_list.append(trajs[tid]["roi_features"])
                traj_durations.append(trajs[tid]["duration"])
        
        return TrajProposal(video_name,cat_ids,traj_boxes,traj_durations,roi_features_list,self.max_proposal)


class Dataset_pku_i3d(Dataset):
    def __init__(self, i3d_dir, dim_i3d, **kargs):
        self.i3d_dir = i3d_dir
        self.dim_i3d = dim_i3d

        super().__init__(**kargs)  # 会调用子类重载的成员函数
    
    def _get_proposal(self,video_name):
        if video_name == "ILSVRC2015_train_00884000":
            video_name = "ILSVRC2015_train_00884000" + "_myFaster18"
        i3d_path = os.path.join(self.i3d_dir,video_name+".npy")
        i3d_features = np.load(i3d_path)
        track_res_path = os.path.join(self.proposal_dir,video_name+".npy") #ILSVRC2015_train_00010001.npy
        track_res = np.load(track_res_path,allow_pickle=True)
        trajs = {box_info[1]:{} for box_info in track_res}
        for tid in trajs.keys():  
            trajs[tid]["frame_ids"] = []
            trajs[tid]["bboxes"] = []
            trajs[tid]["roi_features"] = []
            trajs[tid]["i3d_features"] = []
            trajs[tid]["category_id"] = []   # 如果某个tid只有len==6的box_info，那就无法获取 category_id ，默认为背景

        for idx,box_info in enumerate(track_res):
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 12 + self.dim_boxfeature,"len(box_info)=={}".format(len(box_info))
            
            frame_id = int(box_info[0])
            tid = int(box_info[1])
            tracklet_xywh = box_info[2:6]
            xmin_t,ymin_t,w_t,h_t = tracklet_xywh
            xmax_t = xmin_t + w_t
            ymax_t = ymin_t + h_t
            bbox_t = [xmin_t,ymin_t,xmax_t,ymax_t]
            confidence = box_info[6]
            cat_id = int(box_info[7])
            if cat_id <= 0:
                confidence = 0.0
                bbox = bbox_t + [confidence]
                roi_feature = [0]*self.dim_boxfeature
                i3d_feature = [0]*self.dim_i3d
            else:
                xywh = box_info[8:12]
                xmin,ymin,w,h = xywh
                xmax = xmin+w
                ymax = ymin+h
                bbox = [(xmin+xmin_t)/2, (ymin+ymin_t)/2, (xmax+xmax_t)/2,(ymax+ymax_t)/2,confidence]
                roi_feature = box_info[12:]
                i3d_feature = i3d_features[idx][12:]
                trajs[tid]["category_id"].append(cat_id)

            
            if video_name  == "ILSVRC2015_train_00884000" + "_myFaster18":
                i3d_feature = [0]*self.dim_i3d
            trajs[tid]["bboxes"].append(bbox)
            trajs[tid]["roi_features"].append(roi_feature)
            trajs[tid]["i3d_features"].append(i3d_feature)
            trajs[tid]["frame_ids"].append(frame_id)    # 
    
        # print(video_name)
        for tid in trajs.keys():
            if trajs[tid]["category_id"] == []:
                trajs[tid]["category_id"] = 0
            else:
                # print(video_name,trajs[tid]["category_id"])
                temp = np.argmax(np.bincount(trajs[tid]["category_id"]))  # 求众数
                trajs[tid]["category_id"] = int(temp)
            
            frame_ids = trajs[tid]["frame_ids"]
            start = min(frame_ids)
            end = max(frame_ids) + 1
            dura_len = end - start
            duration = (start,end)  # 前闭后开区间
            roi_feature = np.array(trajs[tid]["roi_features"])
            i3d_feature = np.array(trajs[tid]["i3d_features"]) # list of np.array --> 2d-numpy array
            # concatenate i3d_features and visual_feature
            trajs[tid]["roi_features"] = np.concatenate([roi_feature,i3d_feature],axis=-1)  # shape == (n_frames,2048+832)
            # print(trajs[tid]["box_feats"].shape)
            trajs[tid]["bboxes"] = np.array(trajs[tid]["bboxes"])

            # 将太短的视为背景，后续过滤掉
            if len(frame_ids) < self.min_frames_th:
                trajs[tid]["category_id"] = 0
            else:
                trajs[tid]["duration"] = (start,end)
            
            # 对于非背景的traj， 看是否需要插值
            if trajs[tid]["category_id"] !=0 and len(frame_ids) != dura_len:
                trajs[tid]["roi_features"] = linear_interpolation(trajs[tid]["roi_features"],frame_ids)
                trajs[tid]["bboxes"] = linear_interpolation(trajs[tid]["bboxes"],frame_ids)
            
            if trajs[tid]["category_id"] !=0:
                assert len(trajs[tid]["bboxes"]) == dura_len

        # trajs = {k:v for k,v in trajs.items() if v["category_id"]!=0}
        cat_ids = []
        traj_boxes = []
        roi_features_list = []
        traj_durations = []
        for tid in trajs.keys():
            if trajs[tid]["category_id"] != 0:
                dura_len = trajs[tid]["duration"][1] - trajs[tid]["duration"][0]
                assert len(trajs[tid]["bboxes"]) == dura_len
                cat_ids.append(trajs[tid]["category_id"])
                traj_boxes.append(trajs[tid]["bboxes"])
                roi_features_list.append(trajs[tid]["roi_features"])
                traj_durations.append(trajs[tid]["duration"])
        
        return TrajProposal(video_name,cat_ids,traj_boxes,traj_durations,roi_features_list,self.max_proposal)
    
