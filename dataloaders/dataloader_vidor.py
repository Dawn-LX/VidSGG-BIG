
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import pickle
from copy import deepcopy
# import root_path # uncomment this line when run this .py file (if __name__ == "__main__")
import random
from collections import defaultdict

from utils.utils_func import vIoU, is_overlap, merge_duration_list,VidRead2ImgNpLits
from utils.utils_func import dura_intersection,traj_cutoff,linear_interpolation
from utils.categories_v2 import vidvrd_CatName2Id,vidvrd_PredName2Id,vidor_CatName2Id,vidor_PredName2Id,vidor_CatId2name,vidor_PredId2name,vidor_CatId2FG,vidor_PredId2FG,vidvrd_CatId2name,vidvrd_PredId2name


class TrajProposal(object):
    def __init__(self,video_name,MAX_PROPOSAL,score_th,
        cat_ids,traj_bboxes_with_score,traj_durations,roi_features):
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
        self.scores,self.bboxes_list = self._preprocess_boxes(traj_bboxes_with_score)    # len == num_proposals,# format: xyxy
        # list[float], list[tensor], traj_bboxes.shape == (num_frames,4)
        self.traj_durations = torch.tensor(traj_durations,dtype=torch.long)  # --> shape == (num_proposals, 2)
        self.traj_durations[:,1] -= 1  # 从前开后闭区间 转为 闭区间
        self.features_list = [torch.FloatTensor(r) for r in roi_features]  # len==num_proposals, shape == (num_frames,1024+300) 1024+300=visual_feature+glove_emb
        
        # score_clipping:
        self.scores = torch.FloatTensor(self.scores)
        index = torch.where(self.scores > score_th)[0]
        # assert len(index) > 0
        if len(index) == 0:
            self.num_proposals = 0
            print("video:{} has no proposal after score clipping (score_th={:.2f})".format(video_name,score_th))
            return
        self.scores = self.scores[index]
        self.bboxes_list = [self.bboxes_list[ii] for ii in index]   # format: xyxy
        self.traj_durations = self.traj_durations[index]
        self.features_list = [self.features_list[ii] for ii in index]
        self.cat_ids = self.cat_ids[index]

        # proposal num clipping
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
    def __init__(self,video_info,split,MAX_PREDS,
        traj_cat_ids,traj_durations,traj_bboxes,
        pred_cat_ids,pred_durations,
        adj_matrix_subject,adj_matrix_object):
        self.video_name,self.video_len,self.video_wh = video_info
        self.MAX_PREDS = MAX_PREDS

        assert len(traj_cat_ids) == len(traj_durations)
        assert len(traj_cat_ids) == len(traj_bboxes)
        assert len(pred_cat_ids) == len(pred_durations)
        self.num_trajs = len(traj_cat_ids)
        self.num_preds = len(pred_cat_ids)
        if self.num_trajs == 0:
            print("video:{} : num_trajs=0".format(self.video_name))
            return

        self.traj_cat_ids = torch.LongTensor(traj_cat_ids)     # shape == (num_trajs,)
        self.traj_durations = torch.tensor(traj_durations)     # shape == (num_trajs,2), 
        if len(self.traj_durations.shape) == 1:
            self.traj_durations = self.traj_durations[None,:]
        self.traj_durations[:,1] -= 1                          # convert to closed interval
        self.traj_bboxes = [torch.FloatTensor(b) for b in traj_bboxes] # list[tensor]  len == num_trajs, shape == (num_frames,4),# format: xyxy

        if self.num_preds == 0:
            print("video:{} : num_trajs={},num_preds=0".format(self.video_name,self.num_trajs))
            return 

        self.pred_cat_ids = torch.LongTensor(pred_cat_ids)     # shape == (num_preds,)
        self.pred_durations = torch.tensor(pred_durations).float()     # shape == (num_preds,2),  
        if len(self.pred_durations.shape) == 1:
            self.pred_durations = self.pred_durations[None,:]
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
        if self.num_trajs == 0:
            return self
        self.traj_cat_ids = self.traj_cat_ids.to(device)
        self.traj_durations = self.traj_durations.to(device)
        for i in range(self.num_trajs):
            self.traj_bboxes[i] = self.traj_bboxes[i].to(device)

        if self.num_preds == 0:
            return self
        self.pred_cat_ids = self.pred_cat_ids.to(device)
        self.pred_durations = self.pred_durations.to(device)        # shape == (num_preds,2),  start_fid, end_fid
        self.adj_matrix = self.adj_matrix.to(device)
        
        return self
    
    def to_eval_format(self,enti_FG=False,pred_FG=False):
        if enti_FG:
            entiId2Name = vidor_CatId2FG
        else:
            entiId2Name = vidor_CatId2name #,
        
        if pred_FG:
            predId2Name = vidor_PredId2FG
        else:
            predId2Name = vidor_PredId2name
        
        temp = self.video_name.split('_') 

        assert len(temp) == 2
        video_name = temp[1]  # for vidor dataset

        # assert len(temp) == 1
        # video_name = self.video_name  # for vidvrd dataset
        
        if self.num_trajs==0 or self.num_preds==0:
            return {video_name: []}

        pred_durations = self.pred_durations.clone()
        traj_durations = self.traj_durations.clone()
        pred_durations[:,1] += 1
        traj_durations[:,1] += 1
        
        relations = []
        for p_id in range(self.num_preds):
            s_id = torch.argmax(self.adj_matrix[0,p_id,:]) # subject id   \in 0 ~ num_trajs-1
            o_id = torch.argmax(self.adj_matrix[1,p_id,:]) # object id
            subject_catid = int(self.traj_cat_ids[s_id])
            object_catid  = int(self.traj_cat_ids[o_id])
            pred_catid =    int(self.pred_cat_ids[p_id])

            subject_traj = self.traj_bboxes[s_id]
            # print(type(subject_traj[0][0]))
            object_traj = self.traj_bboxes[o_id]

            subject_dura = traj_durations[s_id].tolist()
            # print(subject_dura)
            object_dura = traj_durations[o_id].tolist()
            pred_dura = (int(pred_durations[p_id,0]),int(pred_durations[p_id,1]))

            duration_so = dura_intersection(subject_dura,object_dura)   # duration intersection of subject and object
            if duration_so == None:
                continue
            else:
                duration_spo = dura_intersection(duration_so,pred_dura) # duration intersection of subject and object and predicate
                if duration_spo == None:
                    duration_spo = duration_so      #TODO  maybe we should discard this (i.e.,maybe we should `continue`)
                else:
                    pass

            subject_traj = traj_cutoff(subject_traj,subject_dura,duration_spo)
            object_traj = traj_cutoff(object_traj,object_dura,duration_spo)
            assert len(subject_traj) == len(object_traj)
            assert len(subject_traj) == duration_spo[1] - duration_spo[0]

            result_per_triplet = dict()
            result_per_triplet["triplet"] = [entiId2Name[subject_catid],predId2Name[pred_catid],entiId2Name[object_catid]]
            result_per_triplet["duration"] = duration_spo   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["sub_traj"] = subject_traj.cpu().numpy().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = object_traj.cpu().numpy().tolist()
            result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"so":duration_so,"p":pred_dura,"spo":duration_spo}
            relations.append(result_per_triplet)
        
        return {video_name: relations}
        
    
    def __repr__(self):
        return "VideoGraph[num_trajs={},num_preds={}]".format(self.num_trajs,self.num_preds)


class Dataset(object):
    def __init__(
        self,
        split,
        video_dir,
        ann_dir,
        proposal_dir,
        classeme_dir,
        max_proposal = 180,
        max_preds = 200,
        score_th = 0.4,
        min_frames_th = 15,
        dim_boxfeature = 1024,
        cache_dir = "datasets/cache",
        cache_tag = "v7_with_clsme",
        video_feature_dir=None
    ):
        self.max_proposal = max_proposal
        self.max_preds = max_preds
        self.score_th = score_th
        self.split = self._get_split(split)  # self.split in ["train","val","test"]
        self.dim_boxfeature = dim_boxfeature
        self.min_frames_th = min_frames_th 
        self.video_dir = video_dir
        self.proposal_dir = proposal_dir  # e.g., proposals/miss60_minscore0p3/VidORval_freq1
        self.classeme_dir = classeme_dir  # e.g., proposals/miss60_minscore0p3/VidORval_freq1_classeme
        self.cache_tag = cache_tag 
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        
        if video_feature_dir is not None:      # e.g., # datasets/vidor_gt_features/
            self.use_video_features = True
        else:
            self.use_video_features = False
        
        
        self.enti_CatName2Id = vidor_CatName2Id
        self.pred_CatName2Id = vidor_PredName2Id
        if self.split == "train":
            self.video_ann_dir = os.path.join(ann_dir,'training')  # e.g., "datasets/vidor-dataset/annotation/training"
        elif self.split == "val":
            self.video_ann_dir = os.path.join(ann_dir,'validation')    # e.g., "datasets/vidor-dataset/annotation/validation"
        else: # self.split == "test"
            self.video_ann_dir = None


        if isinstance(self.proposal_dir,str):
            self.proposal_dir = {0:self.proposal_dir}   # for others
        else:
            assert isinstance(self.proposal_dir,dict)   # for vidor train
        
        self.video_name_list,self.video_name_list_all = self._prepare_video_names()


        data_dict_all = dict()
        for idx,pr_di in enumerate(self.proposal_dir.values()):
            temp = [s for s in pr_di.split('/') if s != ""]
            cache_file = self.cache_tag + '_' + temp[-1] + "_th_{}-{}-{}-{:.2f}".format(self.min_frames_th,self.max_proposal,self.max_preds,self.score_th)
            cache_file = os.path.join(self.cache_dir,cache_file + ".pkl")

            if os.path.exists(cache_file):
                print("loading data into memory ({}/{})..., from cache file {}".format(idx+1,len(self.proposal_dir),cache_file))
                with open(cache_file,'rb') as f:
                    data_dict = pickle.load(f)
            else:
                print("no cache file find, preparing data {}/{}...".format(idx+1,len(self.proposal_dir)))
                print("cache file will be saved as {}".format(cache_file))
                data_dict = dict()
                vid_name_list = os.listdir(pr_di)
                vid_name_list = [v.split('.')[0] for v in vid_name_list]
                for video_name in tqdm(vid_name_list):
                    data_dict[video_name] = self.get_data(video_name)
                with open(cache_file,'wb') as f:
                    pickle.dump(data_dict,f)
                print("data have been saved as cache file: {}".format(cache_file))
            data_dict_all.update(data_dict)
            print("data loaded {}/{}".format(idx+1,len(self.proposal_dir)))

        if self.use_video_features:
            self.video_feature_dict = self.load_video_features(video_feature_dir)    
        
        self.data_dict = data_dict_all
        print("all data have been loaded into memory") 

        ## filter_videos
        if self.split == "train":
            frame_count_th = 2700
            print("filting videos with frame_count_th = {} ...".format(frame_count_th))
            self.video_name_list = [video_name for video_name in self.video_name_list if self.data_dict[video_name][1].video_len <=frame_count_th]
            print("Done. {} videos left".format(len(self.video_name_list)))
        
        # print(list(self.data_dict.keys())[:10])  

    def load_video_features(self,video_feature_dir):
        print("loading video features into memory...")
        video_feature_dict = {}
        for video_name in tqdm(self.video_name_list):
            feature_path = os.path.join(video_feature_dir,video_name+'.npy')
            feature = np.load(feature_path)
            feature = torch.from_numpy(feature)  # shape == (num_clips, 1024)
            
            video_feature_dict[video_name] = feature
            
        
        return video_feature_dict


    def _get_split(self,split):
        train = {x:"train" for x in ["train","training"]}
        val = {x:"val" for x in ["val","validation"]}
        test = {x:"test" for x in ["test","testing"]}
        split_dict = {}
        for x in [train,val,test]:
            split_dict.update(x)
        
        return split_dict[split.lower()]
        
    def _prepare_video_names(self):
        group_list =   os.listdir(self.video_ann_dir)      # e.g., datasets/vidor-dataset/annotation/training
        group_list = sorted(group_list)
        video_name_list_all = []
        for group_name in group_list:
            video_list = os.listdir(os.path.join(self.video_ann_dir,group_name))
            video_list = sorted(video_list)
            video_list = [group_name + "_" + v.split('.')[0] for v in video_list]
            video_name_list_all += video_list
        

        if self.split != "train":
            return video_name_list_all,video_name_list_all

        key_list = list(self.proposal_dir.keys())
        assert "freq1" in self.proposal_dir[key_list[0]]
        v_list = []
        for k in key_list:
            # v_list+= video_name_list_all[k*1000:(k+1)*1000]
            v_list+= video_name_list_all[k*500:(k+1)*500]
        
        return v_list,video_name_list_all
    

    def __getitem__(self,idx):
        video_name = self.video_name_list[idx]
        
        traj_proposal, gt_graph = deepcopy(self.data_dict[video_name]) # for split=="test", gt_graph == None
        if self.use_video_features:
            video_feature = deepcopy(self.video_feature_dict[video_name])
        
        if self.split == "train":
            if traj_proposal.num_proposals == 0 or gt_graph.num_trajs==0 or gt_graph.num_preds==0:
                idx = random.randint(0,len(self.video_name_list)-1)
                return self.__getitem__(idx)
            # elif traj_proposal.video_len > 2000 and self.dataset_type.lower() == "vidor":
            #     # we set frame_count threshold to save gpu memory (to avoid CUDA out-of-memory) when using VidOR-trainset
            #     # frame_count threshold = 2700, videos whose frame_count <= 2700 account for 98% in VidOR-trainset
            #     idx = random.randint(0,len(self.video_name_list)-1)
            #     return self.__getitem__(idx)
            else:
                if self.use_video_features:
                    return video_feature,traj_proposal, gt_graph
                else:
                    return traj_proposal, gt_graph
        else:
            if self.use_video_features:
                return video_feature,traj_proposal, gt_graph
            else:
                return traj_proposal, gt_graph



    def __len__(self):
        return len(self.video_name_list)
    
    def get_data(self,video_name):
        trajs_prop = self._get_proposal(video_name)
        traj_proposal = self.merge_trajs(trajs_prop,None,video_name)
        
        if self.split == "train" or self.split == "val":
            gt_graph = self._get_gt_graph(video_name)
            video_len = gt_graph.video_len    # TODO add traj_proposal.video_len separately and assert `traj_proposal.video_len == gt_graph.video_len`
            video_wh  = gt_graph.video_wh
            # TODO 
            # otherwise we can read the raw video and get its video_len and video_wh (but it will take a lot of time), 
            # and assert `traj_proposal.video_len == gt_graph.video_len`
        else:
            gt_graph = None
            video_len,video_wh = self.get_video_info(video_name)
        
        traj_proposal.video_len = video_len
        traj_proposal.video_wh = video_wh
        
        return traj_proposal, gt_graph
    
    def get_video_info(self,video_name):
        temp = video_name.split('_')
        video_path = os.path.join(self.video_dir,temp[0],temp[1]+".mp4")
        img_list = VidRead2ImgNpLits(video_path)
        video_len = len(img_list)
        x = img_list[0]  # shape == (H,W)
        video_wh = (x.shape[1],x.shape[0])

        return video_len,video_wh



    def videoname2trackres(self,video_name):
        if self.split == "train":
            index = self.video_name_list_all.index(video_name)
            group_index = index//500
            assert "freq1" in self.proposal_dir[group_index]
            # print(video_name,index,group_index)
            track_res_path = os.path.join(self.proposal_dir[group_index],video_name+".npy") #ILSVRC2015_train_00010001.npy
        else:
            assert len(self.proposal_dir) == 1
            track_res_path = os.path.join(self.proposal_dir[0],video_name+".npy") #ILSVRC2015_train_00010001.npy
        
        return track_res_path
    
    def _get_proposal(self,video_name):
        track_res_path = self.videoname2trackres(video_name)
        classeme = np.load(os.path.join(self.classeme_dir,video_name+"_clsme.npy")) # shape == (N,300)
        track_res = np.load(track_res_path,allow_pickle=True)
        # print(classeme.shape,track_res.shape)
        # import time
        # time.sleep(1000)

        trajs = {box_info[1]:{} for box_info in track_res}
        for tid in trajs.keys():  
            trajs[tid]["frame_ids"] = []
            trajs[tid]["bboxes"] = []
            trajs[tid]["box_feats"] = []
            trajs[tid]["classeme"] = []
            trajs[tid]["category_id"] = []   # 如果某个tid只有len==6的box_info，那就无法获取 category_id ，默认为背景

        for idx,box_info in enumerate(track_res):
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
                trajs[tid]["box_feats"].append(roi_feature)
            
            if len(box_info) == 6:
                bbox_t.append(confidence)
                trajs[tid]["bboxes"].append(bbox_t)
                trajs[tid]["box_feats"].append([0]*self.dim_boxfeature)
            else:
                bbox.append(confidence)
                trajs[tid]["bboxes"].append(bbox)
            trajs[tid]["frame_ids"].append(frame_id)    # 
            trajs[tid]["classeme"].append(classeme[idx])  # each shape == (300,)
    

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
            trajs[tid]["bboxes"] = np.array(trajs[tid]["bboxes"])
            visual_feat = np.array(trajs[tid]["box_feats"]) # two-level list --> 2d-numpy array
            classeme_feat = np.array(trajs[tid]["classeme"]) # list of np.array --> 2d-numpy array

            # concatenate classeme and visual_feature
            trajs[tid]["box_feats"] = np.concatenate([visual_feat,classeme_feat],axis=-1)  # shape == (n_frames,1024+300)


            # 将太短的视为背景，后续过滤掉
            if len(frame_ids) < self.min_frames_th:
                trajs[tid]["category_id"] = 0
            else:
                trajs[tid]["duration"] = (start,end)
            
            # 对于非背景的traj， 看是否需要插值
            if trajs[tid]["category_id"] !=0 and len(frame_ids) != dura_len:
                trajs[tid]["box_feats"] = linear_interpolation(trajs[tid]["box_feats"],frame_ids)  # shape == (num_frames,1024+300)
                trajs[tid]["bboxes"] = linear_interpolation(trajs[tid]["bboxes"],frame_ids)              # shape == (num_frames,4)
            
            if trajs[tid]["category_id"] !=0:
                assert len(trajs[tid]["bboxes"]) == dura_len

        trajs = {k:v for k,v in trajs.items() if v["category_id"]!=0}
        return trajs


    def _get_proposal_v2(self,video_name):
        track_res_path = self.videoname2trackres(video_name)
        classeme = np.load(os.path.join(self.classeme_dir,video_name+"_clsme.npy")) # shape == (N,300)
        track_res = np.load(track_res_path,allow_pickle=True)
        # print(classeme.shape,track_res.shape)
        # import time
        # time.sleep(1000)

        trajs = {box_info[1]:{} for box_info in track_res}
        for tid in trajs.keys():  
            trajs[tid]["frame_ids"] = []
            trajs[tid]["bboxes"] = []
            trajs[tid]["box_feats"] = []
            trajs[tid]["classeme"] = []
            trajs[tid]["category_id"] = []   # 如果某个tid只有len==6的box_info，那就无法获取 category_id ，默认为背景

        for idx,box_info in enumerate(track_res):
            if not isinstance(box_info,list):
                box_info = box_info.tolist()
            assert len(box_info) == 6 or len(box_info) == 12 + self.dim_boxfeature,"len(box_info)=={}".format(len(box_info))
            if len(box_info) == 6:
                continue

            frame_id = box_info[0]
            tid = box_info[1]
            tracklet_xywh = box_info[2:6]
            xmin_t,ymin_t,w_t,h_t = tracklet_xywh
            xmax_t = xmin_t + w_t
            ymax_t = ymin_t + h_t
            confidence = box_info[6]
            bbox_t = [xmin_t,ymin_t,xmax_t,ymax_t,confidence]
            cat_id = box_info[7]
            # xywh = box_info[8:12]
            
            roi_feature = box_info[12:]

            trajs[tid]["bboxes"].append(bbox_t)
            trajs[tid]["category_id"].append(cat_id)
            trajs[tid]["box_feats"].append(roi_feature)
            
            trajs[tid]["frame_ids"].append(frame_id)    # 
            trajs[tid]["classeme"].append(classeme[idx])  # each shape == (300,)
    

        for tid in trajs.keys():
            if trajs[tid]["category_id"] == []:
                trajs[tid]["category_id"] = 0
            else:
                # print(trajs[tid]["category_id"])
                temp = np.argmax(np.bincount(trajs[tid]["category_id"]))  # 求众数
                trajs[tid]["category_id"] = int(temp)

            

            # 将太短的视为背景，后续过滤掉
            frame_ids = trajs[tid]["frame_ids"]
            if len(frame_ids) < self.min_frames_th:
                trajs[tid]["category_id"] = 0
            else:
                start = min(frame_ids)
                end = max(frame_ids) + 1
                dura_len = end - start
                trajs[tid]["duration"] = (start,end)
                trajs[tid]["bboxes"] = np.array(trajs[tid]["bboxes"])
                visual_feat = np.array(trajs[tid]["box_feats"]) # two-level list --> 2d-numpy array
                classeme_feat = np.array(trajs[tid]["classeme"]) # list of np.array --> 2d-numpy array

                # concatenate classeme and visual_feature
                trajs[tid]["box_feats"] = np.concatenate([visual_feat,classeme_feat],axis=-1)  # shape == (n_frames,1024+300)
                
            
            # 对于非背景的traj， 看是否需要插值
            if trajs[tid]["category_id"] !=0 and len(frame_ids) != dura_len:
                trajs[tid]["box_feats"] = linear_interpolation(trajs[tid]["box_feats"],frame_ids)  # shape == (num_frames,1024+300)
                trajs[tid]["bboxes"] = linear_interpolation(trajs[tid]["bboxes"],frame_ids)              # shape == (num_frames,4)
            
            if trajs[tid]["category_id"] !=0:
                assert len(trajs[tid]["bboxes"]) == dura_len

        trajs = {k:v for k,v in trajs.items() if v["category_id"]!=0}
        tid_list = list(trajs.keys())

        cat_ids = []
        traj_boxes = []
        roi_features_list = []
        traj_durations = []
        for tid in tid_list:
            assert trajs[tid]["category_id"] != 0
            dura_len = trajs[tid]["duration"][1] - trajs[tid]["duration"][0]
            assert len(trajs[tid]["bboxes"]) == dura_len
            cat_ids.append(trajs[tid]["category_id"])
            traj_boxes.append(trajs[tid]["bboxes"])
            roi_features_list.append(trajs[tid]["box_feats"])
            traj_durations.append(trajs[tid]["duration"])
        
        return TrajProposal(video_name,self.max_proposal,self.score_th,cat_ids,traj_boxes,traj_durations,roi_features_list)
    

    def _get_gttraj(self,video_name):
        """format
        trajs_dict = {
            tid_0: traj_dict_0,
            ...
            tid_x: traj_dict_x,
        }

        # tid_x is int, stands for trajectory id
        # traj_dict_x is a dich, which has the following format:
        traj_dict_x = {
            "category_id": int,         # range 1 ~ num_classes - 1 (num_classes including background, whose idx=0, gt_trajs are guaranteed not background)        
            "frame_ids":frame_ids       # np.ndarray, shape == (n_frames,)
            "bboxes":bboxes             # np.ndarray, shape == (n_frames,4)
            "box_feats":box_feats       # np.ndarray, shape == (n_frames,2048)
        }
        """
        trajs = self.gt_trajswithfeatures[video_name]
        for tid in trajs.keys():
            frame_ids = trajs[tid]["frame_ids"]
            start = min(frame_ids)
            end = max(frame_ids) + 1
            dura_len = end - start
            duration = (start,end)  # 前闭后开区间
            trajs[tid]["duration"] = duration
            scores = np.ones(shape=(len(frame_ids),1))
            trajs[tid]["bboxes"] = np.concatenate([trajs[tid]["bboxes"],scores],axis=1)
            assert trajs[tid]["bboxes"].shape == (len(frame_ids),5)
            if len(frame_ids) != dura_len:
                trajs[tid]["box_feats"] = linear_interpolation(trajs[tid]["box_feats"],frame_ids)  # shape == (num_frames,2048)
                trajs[tid]["bboxes"] = linear_interpolation(trajs[tid]["bboxes"],frame_ids)              # shape == (num_frames,4)

        return trajs
    
    def merge_trajs(self,trajs_prop,trajs_gt,video_name):
        trajs = {}
        trajs.update(trajs_prop)
        if trajs_gt != None:
            max_tid = max(list(trajs_prop.keys())) if list(trajs_prop.keys()) else 0
            max_tid += 1
            trajs_gt = {tid+max_tid:v for tid,v in trajs_gt.items()}
            trajs.update(trajs_gt)
        tid_list = list(trajs.keys())
        random.shuffle(tid_list)  # not necessary 

        cat_ids = []
        traj_boxes = []
        roi_features_list = []
        traj_durations = []
        for tid in tid_list:
            assert trajs[tid]["category_id"] != 0
            dura_len = trajs[tid]["duration"][1] - trajs[tid]["duration"][0]
            assert len(trajs[tid]["bboxes"]) == dura_len
            cat_ids.append(trajs[tid]["category_id"])
            traj_boxes.append(trajs[tid]["bboxes"])
            roi_features_list.append(trajs[tid]["box_feats"])
            traj_durations.append(trajs[tid]["duration"])
        
        return TrajProposal(video_name,self.max_proposal,self.score_th,cat_ids,traj_boxes,traj_durations,roi_features_list)
    

    def _get_gt_graph(self,video_name):
        
        temp = video_name.split('_')
        anno_name = temp[0] + "/" + temp[1] + ".json"  # e.g., "ILSVRC2015_train_00005003.json" or "0004/11566980553.json"
        
        video_ann_path = os.path.join(self.video_ann_dir, anno_name) # datasets/vidor-dataset/annotation/validation

        ## 1. construct trajectory annotations from frame-level bbox annos
        if os.path.exists(video_ann_path):   
            with open(video_ann_path,'r') as f:
                video_anno = json.load(f)
        else:
            print(video_name,"not find its anno")
            raise NotImplementedError
        
        video_len = len(video_anno["trajectories"])
        video_wh = (video_anno["width"],video_anno["height"])

        traj_categories = video_anno["subject/objects"]      # tid 未必从 0 ~ len(traj_categories)-1 都有
        # tid2category_map = [traj["category"] for traj in traj_categories] #  这样写是不对的, tid 未必从 0 ~ len(traj_categories)-1 都有
        tid2category_map = {traj["tid"]:traj["category"] for traj in traj_categories} # 要这样搞
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
                category_id = self.enti_CatName2Id[category_name]

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
        for idx,tid in enumerate(trajs.keys()):   # NOTE python 3.6 及以后， 字典这样遍历是有序的，也就是说，每次跑这个循环，idx和tid的对应关系都是不变的
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
            pred_catid = self.pred_CatName2Id[pred_name]

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

        return VideoGraph(video_info,self.split,self.max_preds,
                traj_cat_ids,traj_durations,traj_bbox_list,
                pred_cat_ids,pred_durations,
                adj_matrix_subject,adj_matrix_object
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

            batch_proposal = [b[0] for b in batch]
            batch_gt_graph = [b[1] for b in batch]

            return batch_proposal,batch_gt_graph
        
        def collator_func_v3(batch):
            
            batch_video_feature = [b[0] for b in batch]
            batch_proposal = [b[1] for b in batch]
            batch_gt_graph = [b[2] for b in batch]
            

            return batch_video_feature,batch_proposal,batch_gt_graph

        if self.use_video_features:
            return collator_func_v3
        else:
            return collator_func_v2


if __name__ == "__main__":
    import torch
    import torch.utils.data
    # datasets/vidvrd-dataset/test/ILSVRC2015_train_00219001.json
    video_name = "ILSVRC2015_train_00219001"
    dataset_config = dict(
        frame_img_path = "datasets/vidvrd-dataset/images",
        ann_path = "datasets/vidvrd-dataset",
        test_proposal_dir = "proposals/vidvrd-dataset/miss0_minscore0p3/VidVRD_test_every1frames",
        train_proposal_dir = "proposals/vidvrd-dataset/miss0_minscore0p3/VidVRD_train_every1frames",
        min_frames_th = 5
    )
    dataset = Dataset(**dataset_config,is_train=False,in_memory_mode=False)
    
    traj_proposal, gt_graph = dataset.get_data(video_name)
    print(traj_proposal)
    print(gt_graph)
    gt = gt_graph.to_eval_format()
    gt = list(gt.values())[0]
    print(gt[2])
    print(gt[8])
