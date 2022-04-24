import os
import torch

from VidVRDhelperEvalAPIs import eval_visual_relation

from .utils_func import create_logger,load_json
from .utils_func import traj_cutoff,dura_intersection,dura_intersection_ts
from .categories_v2 import vidvrd_CatId2name,vidvrd_PredId2name,vidor_CatId2name,vidor_PredId2name,PKU_vidvrd_CatId2name



class EvalFmtCvtor(object):
    def __init__(self,dataset_type):
        self.dataset_type = dataset_type.lower()
        if self.dataset_type == "vidvrd":
            self.entiId2Name = vidvrd_CatId2name
            self.predId2Name = vidvrd_PredId2name
        elif self.dataset_type == "vidor":
            self.entiId2Name = vidor_CatId2name 
            self.predId2Name = vidor_PredId2name
        else:
            assert False

    
    def _reset_video_name(self,video_name):
        if self.dataset_type == "vidor":
            temp = video_name.split('_')  # e.g., "0001_3598080384"
            assert len(temp) == 2
            video_name = temp[1]
        elif self.dataset_type == "vidvrd":
            # e.g., video_name == "ILSVRC2015_train_00005015"
            pass
        else:
            assert False
        
        return video_name
    
 

    def prepare_gt(self,gt_graph):
        
        traj_cat_ids = gt_graph.traj_cat_ids  # shape == (n_traj,)
        traj_duras = gt_graph.traj_durations  # shape == (n_traj,2)
        pred_durations = gt_graph.pred_durations.type(torch.long)  # shape == (n_gt_pred,2)
        pred_cat_ids  = gt_graph.pred_cat_ids     # shape == (n_gt_pred,2)


        
        pred2so_ids = torch.argmax(gt_graph.adj_matrix,dim=-1).t()  # enti index,  shape == (n_gt_pred,2)
        pred2so_catids = traj_cat_ids[pred2so_ids] # shape == (n_gt_pred,2)
        quintuples = torch.cat([pred_cat_ids[:,None],pred2so_catids,pred2so_ids],dim=-1)  
        # shape == (n_gt_pred,5) format: [pred_catid,subj_catid,obj_catid,subj_id,obj_id]


        dura_inters,dura_mask = dura_intersection_ts(traj_duras,traj_duras)  # shape == (n_traj,n_traj,2)
        dura_inters = dura_inters[pred2so_ids[:,0],pred2so_ids[:,1],:] # shape == (n_gt_pred,2)

        check_inter,mask_ = dura_intersection_ts(pred_durations,dura_inters,broadcast=False)  # shape == (n_gt_pred,2)
        assert torch.all(check_inter == pred_durations)  #NOTE you can check this. all pred_durations are within subj&obj overlaps

        gt_info = (
            quintuples,
            pred_durations
        )
        
        
        return gt_info



    
    def to_eval_format_pr(self,proposal,pr_triplet,preserve_debug_info=False,use_pku=False):
        '''
        this func is compatible for predictions both before and after grounding
        use_pku: Liu et al, (Peking University), paper: "Beyond Short-Term Snippet: Video Relation Detection with Spatio-Temporal Global Context"
        they have a different id2category map 
        '''
        if use_pku:
            pr_entiId2Name = PKU_vidvrd_CatId2name
        else:
            pr_entiId2Name = self.entiId2Name

        video_name = self._reset_video_name(proposal.video_name)
        if pr_triplet is None:
            return {video_name : []}
        
        traj_bboxes = proposal.bboxes_list       # list[tensor], len==num_proposals, shape==(n_frames,4)
        durations_list = proposal.traj_durations.clone() # shape == (num_proposals,2)
        durations_list = durations_list.tolist() 

        (  
            pr_qtuple,      # shape == (n_pr,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
            pr_score,       # shape == (n_pr,)
            inter_dura,      # shape == (n_pr,2) 
            # debug_info
        ) = pr_triplet

        n_pr,_ = pr_qtuple.shape
        # pred_duras_float = debug_info
        if isinstance(pr_qtuple,torch.Tensor):
            pr_qtuple = pr_qtuple.tolist()
        if isinstance(pr_score,torch.Tensor):
            pr_score = pr_score.tolist()
        if isinstance(inter_dura,torch.Tensor):
            inter_dura = inter_dura.tolist()
        
        results_per_video = []
        for p_id in range(n_pr):
            pred_catid,subj_catid,obj_catid,subj_tid,obj_tid = pr_qtuple[p_id]
            if pred_catid == 0:
                continue
            
            ori_sub_traj = traj_bboxes[subj_tid]
            ori_obj_traj = traj_bboxes[obj_tid]

            dura_ = (inter_dura[p_id][0],inter_dura[p_id][1]+1)
            
            subject_dura_ = durations_list[subj_tid]
            subject_dura = (subject_dura_[0],subject_dura_[1]+1) # 转为前闭后开区间
            object_dura_ = durations_list[obj_tid]
            object_dura = (object_dura_[0],object_dura_[1]+1)
            # print(subject_dura,dura_,"subject_dura,dura_")
            subject_traj = traj_cutoff(ori_sub_traj,subject_dura,dura_,video_name)
            object_traj = traj_cutoff(ori_obj_traj,object_dura,dura_,video_name)
            assert len(subject_traj) == len(object_traj)
            assert len(subject_traj) == dura_[1] - dura_[0]

            # pr_float_dura = pred_duras_float[p_id,:].tolist()
            
            result_per_triplet = dict()
            result_per_triplet["triplet"] = [pr_entiId2Name[subj_catid],self.predId2Name[pred_catid],pr_entiId2Name[obj_catid]]
            result_per_triplet["duration"] = dura_   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["score"] = float(pr_score[p_id])
            result_per_triplet["sub_traj"] = subject_traj.cpu().numpy().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = object_traj.cpu().numpy().tolist()
            
            ################## for debug #################
            if preserve_debug_info:
                result_per_triplet["triplet_tid"] = (int(subj_tid),int(pred_catid),int(obj_tid))  # 如果用 [s_id,p_id,p_catid,o_id]的话，那肯定是唯一的
                result_per_triplet["ori_sub_traj"] = ori_sub_traj.cpu().numpy().tolist()     # len == duration_spo[1] - duration_spo[0]
                result_per_triplet["ori_obj_traj"] = ori_obj_traj.cpu().numpy().tolist()
                # result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"inter":dura_,"pr_float_dura":pr_float_dura}
                result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"inter":dura_}
            ################## for debug #################
            
            results_per_video.append(result_per_triplet)
        

        # results_per_video = sorted(results_per_video,key=lambda x: x["score"],reverse=True)  # large --> small
        # results_per_video = results_per_video[:100]
        return {video_name : results_per_video}


    
    def to_eval_format_pr_wo_cutoff(self,proposal,pr_triplet,preserve_debug_info=False,use_pku=False):
        '''
        this func is without traj_cutoff, this cause bad performance
        '''
        if use_pku:
            pr_entiId2Name = PKU_vidvrd_CatId2name
        else:
            pr_entiId2Name = self.entiId2Name

        video_name = self._reset_video_name(proposal.video_name)
        if pr_triplet is None:
            return {video_name : []}
        
        traj_bboxes = proposal.bboxes_list       # list[tensor], len==num_proposals, shape==(n_frames,4)
        durations_list = proposal.traj_durations.clone() # shape == (num_proposals,2)
        durations_list = durations_list.tolist() 

        (  
            pr_qtuple,      # shape == (n_pr,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
            pr_score,       # shape == (n_pr,)
            inter_dura,      # shape == (n_pr,2) 
            # debug_info
        ) = pr_triplet

        n_pr,_ = pr_qtuple.shape
        # pred_duras_float = debug_info
        if isinstance(pr_qtuple,torch.Tensor):
            pr_qtuple = pr_qtuple.tolist()
        if isinstance(pr_score,torch.Tensor):
            pr_score = pr_score.tolist()
        if isinstance(inter_dura,torch.Tensor):
            inter_dura = inter_dura.tolist()
        
        results_per_video = []
        for p_id in range(n_pr):
            pred_catid,subj_catid,obj_catid,subj_tid,obj_tid = pr_qtuple[p_id]
            if pred_catid == 0:
                continue
            
            ori_sub_traj = traj_bboxes[subj_tid]
            ori_obj_traj = traj_bboxes[obj_tid]

            dura_ = (inter_dura[p_id][0],inter_dura[p_id][1]+1)
            
            subject_dura_ = durations_list[subj_tid]
            subject_dura = (subject_dura_[0],subject_dura_[1]+1) # 转为前闭后开区间
            object_dura_ = durations_list[obj_tid]
            object_dura = (object_dura_[0],object_dura_[1]+1)
            # print(subject_dura,dura_,"subject_dura,dura_")
            subject_traj = traj_cutoff(ori_sub_traj,subject_dura,dura_,video_name)
            object_traj = traj_cutoff(ori_obj_traj,object_dura,dura_,video_name)
            assert len(subject_traj) == len(object_traj)
            assert len(subject_traj) == dura_[1] - dura_[0]

            # pr_float_dura = pred_duras_float[p_id,:].tolist()
            
            result_per_triplet = dict()
            result_per_triplet["triplet"] = [pr_entiId2Name[subj_catid],self.predId2Name[pred_catid],pr_entiId2Name[obj_catid]]
            result_per_triplet["duration"] = dura_   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["score"] = float(pr_score[p_id])
            result_per_triplet["sub_traj"] = ori_sub_traj.cpu().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = ori_obj_traj.cpu().tolist()
            
            ################## for debug #################
            if preserve_debug_info:
                result_per_triplet["triplet_tid"] = (int(subj_tid),int(pred_catid),int(obj_tid))  # 如果用 [s_id,p_id,p_catid,o_id]的话，那肯定是唯一的
                # result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"inter":dura_,"pr_float_dura":pr_float_dura}
                result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"inter":dura_}
            ################## for debug #################
            
            results_per_video.append(result_per_triplet)
        

        # results_per_video = sorted(results_per_video,key=lambda x: x["score"],reverse=True)  # large --> small
        # results_per_video = results_per_video[:100]
        return {video_name : results_per_video}
    


    def to_eval_format_gt(self,gt_graph):
        video_name = self._reset_video_name(gt_graph.video_name)

        if gt_graph.num_trajs==0 or gt_graph.num_preds==0:
            return {video_name: []}

        gt_info = self.prepare_gt(gt_graph)

        traj_bboxes = gt_graph.traj_bboxes  # list[tensor],each shape == (n_frames,4) # format: xyxy
        traj_durations = gt_graph.traj_durations.clone().tolist()

        (
            quintuples,     # shape == (n_gt_pred,5) format: [pred_catid,subj_catid,obj_catid,subj_id,obj_id]
            inter_dura      # shape == (n_gt_pred,2)
        ) = gt_info

        n_gt,_ = quintuples.shape
        if n_gt == 0:
            return {video_name:[]}
        
        if isinstance(quintuples,torch.Tensor):
            quintuples = quintuples.tolist()
        if isinstance(inter_dura,torch.Tensor):
            inter_dura = inter_dura.tolist()
        

        results_per_video = []
        for g_id in range(n_gt):
            pred_catid,subj_catid,obj_catid,subj_tid,obj_tid = quintuples[g_id]
            if pred_catid == 0:
                continue
            
            subject_traj = traj_bboxes[subj_tid]
            object_traj = traj_bboxes[obj_tid]
            subject_dura = (traj_durations[subj_tid][0],traj_durations[subj_tid][1]+1)
            object_dura = (traj_durations[obj_tid][0],traj_durations[obj_tid][1]+1)

            dura_spo = (inter_dura[g_id][0],inter_dura[g_id][1]+1)

            subject_traj = traj_cutoff(subject_traj,subject_dura,dura_spo)
            object_traj = traj_cutoff(object_traj,object_dura,dura_spo)
            assert len(subject_traj) == len(object_traj)
            assert len(subject_traj) == dura_spo[1] - dura_spo[0]

            result_per_triplet = dict()
            result_per_triplet["triplet"] = [self.entiId2Name[subj_catid],self.predId2Name[pred_catid],self.entiId2Name[obj_catid]]
            result_per_triplet["duration"] = dura_spo   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["sub_traj"] = subject_traj.cpu().numpy().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = object_traj.cpu().numpy().tolist()
            result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"inter":dura_spo}
            results_per_video.append(result_per_triplet)
        
        return {video_name : results_per_video}
            
    
    def to_eval_format_gt_old(self,gt_graph):
        video_name = self._reset_video_name(gt_graph.video_name)

        if gt_graph.num_trajs==0 or gt_graph.num_preds==0:
            return {video_name: []}

        pred_durations = gt_graph.pred_durations.clone()
        traj_durations = gt_graph.traj_durations.clone()
        pred_durations[:,1] += 1
        traj_durations[:,1] += 1
        
        relations = []
        for p_id in range(gt_graph.num_preds):
            s_id = torch.argmax(gt_graph.adj_matrix[0,p_id,:]) # subject id   \in 0 ~ num_trajs-1
            o_id = torch.argmax(gt_graph.adj_matrix[1,p_id,:]) # object id
            subject_catid = int(gt_graph.traj_cat_ids[s_id])
            object_catid  = int(gt_graph.traj_cat_ids[o_id])
            pred_catid =    int(gt_graph.pred_cat_ids[p_id])

            subject_traj = gt_graph.traj_bboxes[s_id]
            # print(type(subject_traj[0][0]))
            object_traj = gt_graph.traj_bboxes[o_id]

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
            result_per_triplet["triplet"] = [self.entiId2Name[subject_catid],self.predId2Name[pred_catid],self.entiId2Name[object_catid]]
            result_per_triplet["duration"] = duration_spo   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["sub_traj"] = subject_traj.cpu().numpy().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = object_traj.cpu().numpy().tolist()
            result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"so":duration_so,"p":pred_dura,"spo":duration_spo}
            relations.append(result_per_triplet)
        
        return {video_name: relations}

