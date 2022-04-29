import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_func import dura_intersection_ts,unique_with_idx_nd,stack_with_padding,stack_with_repeat_2d

class Base_C(nn.Module): 
    def __init__(self,config,is_train=True):
        super().__init__()
        self.is_train = is_train

        ## 1. configs
        # 1.1 model configs
        self.num_pred_cats = config["num_pred_cats"]
        self.num_enti_cats = config["num_enti_cats"]
        self.dim_feat = config["dim_feat"]          # 2048 or 1024 (dimension of each bbox's RoI feature, depend on the detector)
        self.dim_clsme = config["dim_clsme"]
        self.dim_enti = config["dim_enti"]
        self.dim_ffn  = config["dim_ffn"]
        self.enco_pool_len = config["enco_pool_len"]        
        self.bias_matrix_path = config["bias_matrix_path"]
        self.EntiNameEmb_path = config.get("EntiNameEmb_path",None)
        self.use_clsme = config["use_clsme"]
        self.rt_triplets_topk = config["rt_triplets_topk"]

        if self.EntiNameEmb_path is None:  # for trajs that have classseme feature
            self.EntiNameEmb = None
        else:                              # otherwise, use category of traj to get a classeme feature
            EntiNameEmb = np.load(self.EntiNameEmb_path)
            EntiNameEmb = torch.from_numpy(EntiNameEmb).float()
            self.EntiNameEmb = nn.Parameter(EntiNameEmb, requires_grad = False)  
            # shape == (num_enti_cats,dim_emb) == (81,300) or (36, 300) # including background
            assert self.EntiNameEmb.shape == (self.num_enti_cats,self.dim_clsme)
        
        
        bias_matrix = np.load(self.bias_matrix_path)
        bias_matrix = torch.from_numpy(bias_matrix).float()
        self.bias_matrix = nn.Parameter(bias_matrix, requires_grad = True)  # shape == (num_enti_cats,num_enti_cats,num_pred_cats) # (81,81,51), including background
        assert self.bias_matrix.shape == (self.num_enti_cats,self.num_enti_cats,self.num_pred_cats)


        ## 3. layers for entity features initialization
        self.fc_feat2enti = nn.Sequential(
            nn.Linear(self.dim_feat,self.dim_enti),
            nn.ReLU(),
            nn.Linear(self.dim_enti,self.dim_enti),
            nn.ReLU()
        )
        self.fc_bbox2enti = nn.Sequential(
            nn.Linear(8,self.dim_enti),
            nn.ReLU(),
            nn.Linear(self.dim_enti,self.dim_enti),
            nn.ReLU()
        )
        self.conv_feat2enti = nn.Conv1d(self.dim_enti*2,self.dim_enti,kernel_size=3,padding=1,stride=2)

        self.fc_enti2enco = nn.Sequential(
            nn.Linear(self.dim_enti*self.enco_pool_len,self.dim_enti),
            nn.ReLU(),
            nn.Linear(self.dim_enti,self.dim_enti),
            nn.ReLU()
        )

        ## 5. layers for classification 
        self.fc_pred2logits = nn.Sequential(
            nn.Linear(self.dim_clsme*2 + self.dim_enti*2, self.dim_ffn),
            nn.ReLU(),
            nn.Linear(self.dim_ffn,self.num_pred_cats)
        )
        

        self._reset_parameters()

    
    def _reset_parameters(self):
        skip_init_param_names = [
            "bias_matrix",
            "EntiNameEmb",
        ]
        for name,p in self.named_parameters():
            if name in skip_init_param_names:  
                print("skip init param: {}".format(name))
                continue

            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
            if "bias" in name:
                nn.init.zeros_(p)
    
    def forward(self,proposal_list,pos_id_list=None,label_list=None,topk=10):

        if self.is_train:
            assert pos_id_list is not None 
            assert label_list is not None

            return self._forward_train(proposal_list,pos_id_list,label_list)
        else: 
            self.topk = topk
            return self._forward_test(proposal_list)


    def trajid2pairid(self,num_prop):
        mask = torch.ones(size=(num_prop,num_prop),dtype=torch.bool)
        mask[range(num_prop),range(num_prop)] = 0
        # print(mask)
        pair_ids = mask.nonzero(as_tuple=False)
        # print(pair_ids)

        return pair_ids

    def _forward_test(self,proposal_list):

        triplets = []
        for ii,proposal in enumerate(proposal_list):
            if proposal.num_proposals == 0:  # train 的时候 num_proposal == 0 的会被过滤掉
                triplets.append(None)
            else:
                pair_ids = self.trajid2pairid(proposal.num_proposals)
                pair_ids = pair_ids.to(proposal.traj_durations.device)
                pred_logits = self.forward_propagation(proposal,pair_ids)

                ret = self.construct_triplet(proposal,pred_logits,pair_ids)
                triplets.append(ret)

        
        return triplets


    def _preprocess_proposal(self,proposal):
        video_len,video_wh = proposal.video_len, proposal.video_wh
        w,h = video_wh
        traj_durations,bboxes_list,feature_list = proposal.traj_durations,proposal.bboxes_list,proposal.features_list
        # traj_durations: shape == (n_trajs,2)
        # bboxes_list： list[tensor], len == n_trajs, shape == (num_frames,4)
        # features_list # list[tensor], len==n_trajs, shape == (num_frames,1024)
        traj_bboxes = []
        traj_features = []
        for pid in range(len(bboxes_list)):
            bboxes = bboxes_list[pid].clone()  # shape == (num_frames, 4) tensor
            bboxes[:,0:4:2] /= w  
            bboxes[:,1:4:2] /= h 

            bbox_ctx = (bboxes[:,2] + bboxes[:,0])/2
            bbox_cty = (bboxes[:,3] + bboxes[:,1])/2
            bbox_w = bboxes[:,2] - bboxes[:,0]
            bbox_h = bboxes[:,3] - bboxes[:,1]

            diff_ctx = bbox_ctx[1:] - bbox_ctx[:-1]
            diff_cty = bbox_cty[1:] - bbox_cty[:-1]
            diff_w = bbox_w[1:] - bbox_w[:-1]
            diff_h = bbox_h[1:] - bbox_h[:-1]
            bbox_feat = [
                bbox_ctx,diff_ctx,
                bbox_cty,diff_cty,
                bbox_w,diff_w,
                bbox_h,diff_h
            ]
            bbox_feat = stack_with_padding(bbox_feat,dim=1) # shape == (n_frames,8)
            traj_bboxes.append(bbox_feat)

            features = feature_list[pid]
            traj_features.append(features)

        
        traj_bboxes = stack_with_repeat_2d(traj_bboxes,dim=0)     # shape == (n_trajs,max_frame, 8)
        traj_features = stack_with_repeat_2d(traj_features,dim=0) # shape == (n_trajs, max_frame, 1024)

        return traj_bboxes,traj_features,traj_durations



    def forward_propagation(self,proposal,pairid2trajids):
        n_trajs,video_len = proposal.num_proposals,proposal.video_len
        traj_bboxes,traj_features,traj_dura = self._preprocess_proposal(proposal) 
        # shape == (n_trajs, max_frames,4), (n_trajs, max_frames,1024+300), (n_trajs, 2)
        traj_visual  = traj_features[:,:,:self.dim_feat]
        assert traj_visual.shape[2] == self.dim_feat
        if self.use_clsme:
            traj_classeme = traj_features[:,:,self.dim_feat:]  # shape == (n_trajs, max_frames,dim_clsme)
            if self.EntiNameEmb is None:  # i.e., if we don't use category embedding as classeme feature
                assert traj_classeme.shape[2] == self.dim_clsme
        
        
        traj_bboxes =   self.fc_bbox2enti(traj_bboxes) 
        traj_visual = self.fc_feat2enti(traj_visual)
        enti_features = torch.cat([traj_bboxes,traj_visual],dim=-1)
        
        enti_features = enti_features.permute(0,2,1) # shape == (n_trajs, dim_state, max_frames)
        enti_nodes = self.conv_feat2enti(enti_features) # shape == (n_trajs, dim_state, max_frames//2) 
        enti_nodes = enti_nodes.permute(0,2,1) # shape == (n_trajs,  max_frames//2, dim_state)


        ## encode 
        enti2enco = enti_nodes.permute(0,2,1) # shape == (n_trajs, dim_state, max_frames//2)
        enti2enco = F.adaptive_max_pool1d(enti2enco,output_size=self.enco_pool_len) # shape == (n_trajs, dim_state, pool_outlen)  
        enti2enco = enti2enco.reshape(n_trajs,-1)  # shape == (n_trajs, dim_state*pool_outlen)
        enti2enco = self.fc_enti2enco(enti2enco)    # shape == (n_trajs, dim_state)
        
        if self.use_clsme and (self.EntiNameEmb is None):
            traj_clsme_avg = traj_classeme.mean(dim=1) # shape == (n_enti,300)
        else:
            traj_clsme_avg = None
        
        pred_logits = self.prediction_head(pairid2trajids,proposal.cat_ids,traj_clsme_avg,enti2enco,proposal.video_name) # shpae == (n_querys, n_pred_cat)

        
        return pred_logits
    

    def extract_traj_features(self,proposal):
        n_trajs,video_len = proposal.num_proposals,proposal.video_len
        traj_bboxes,traj_features,traj_dura = self._preprocess_proposal(proposal) 
        # shape == (n_trajs, max_frames,4), (n_trajs, max_frames,1024+300), (n_trajs, 2)
        traj_visual  = traj_features[:,:,:self.dim_feat]
        traj_classeme = traj_features[:,:,self.dim_feat:]  # shape == (n_trajs, max_frames,dim_clsme)
        if self.EntiNameEmb is None:
            assert traj_visual.shape[2] == self.dim_feat and traj_classeme.shape[2] == self.dim_clsme
        else:
            assert traj_visual.shape[2] == self.dim_feat #and traj_classeme.shape[2] == 0
            traj_classeme = None

        traj_bboxes =   self.fc_bbox2enti(traj_bboxes) 
        traj_visual = self.fc_feat2enti(traj_visual)
        enti_features = torch.cat([traj_bboxes,traj_visual],dim=-1)
        
        enti_features = enti_features.permute(0,2,1) # shape == (n_trajs, dim_state, max_frames)
        enti_nodes = self.conv_feat2enti(enti_features) # shape == (n_trajs, dim_state, max_frames//2) 
        enti_nodes = enti_nodes.permute(0,2,1) # shape == (n_trajs,  max_frames//2, dim_state)


        ## encode 
        enti2enco = enti_nodes.permute(0,2,1) # shape == (n_trajs, dim_state, max_frames//2)
        enti2enco = F.adaptive_max_pool1d(enti2enco,output_size=self.enco_pool_len) # shape == (n_trajs, dim_state, pool_outlen)  
        enti2enco = enti2enco.reshape(n_trajs,-1)  # shape == (n_trajs, dim_state*pool_outlen)
        enti2enco = self.fc_enti2enco(enti2enco)    # shape == (n_trajs, dim_state)

        return enti2enco,traj_dura
        


    def prediction_head(self,pairid2trajids,cat_ids,enti_clsme,enti_feat,video_name):
        # pairid2trajids  (n_pos_pairs,2)

        # cat_ids: shape == (n_enti,)
        # enti_clsme.shape ==(n_enti,300)
        # self.bias_matrix shape == (n_enti_cat,n_enti_cat,n_pred_cat) # (81,81,51), including background

        pred_socatid = cat_ids[pairid2trajids]  # enti categories, shape == (n_pos_pairs,2)
        pred_bias = self.bias_matrix[pred_socatid[:,0],pred_socatid[:,1],:] # shape == (n_querys,n_pred_cat)
        sub_feat = enti_feat[pairid2trajids[:,0],:]  # shape == (n_pos_pairs, dim_enti)
        obj_feat = enti_feat[pairid2trajids[:,1],:]  # shape == (n_pos_pairs, din_enti)

        if self.use_clsme:
            if self.EntiNameEmb is None:
                assert enti_clsme is not None
                sub_clsme = enti_clsme[pairid2trajids[:,0],:]  # shape == (n_pos_pairs, 300)
                obj_clsme = enti_clsme[pairid2trajids[:,1],:]  # shape == (n_pos_pairs, 300)
            else:
                assert enti_clsme is None
                sub_clsme = self.EntiNameEmb[pred_socatid[:,0],:]  # shape == (n_pos_pairs, 300)
                obj_clsme = self.EntiNameEmb[pred_socatid[:,1],:]  # shape == (n_pos_pairs, 300)
            # for x in [sub_clsme,obj_clsme,sub_feat,obj_feat]:
            #     print(x.shape)
            combined_features = torch.cat([sub_clsme,obj_clsme,sub_feat,obj_feat],dim=-1)  # shape == (n_querys,600+2*dim_enti)
        else:
            combined_features = torch.cat([sub_feat,obj_feat],dim=-1)  # shape == (n_querys,2*dim_enti)
                
        pred_logits = self.fc_pred2logits(combined_features)
        pred_logits = pred_logits + pred_bias
           
        return pred_logits
  
  
    def _forward_train(self,proposal_list,pos_id_list,label_list):
        
        # pos_id_list = [pairid2trajids, pairid2trajids,...]
        # label_list = [multihot,multihot,...]

        # pairid2trajids.shape == (n_pos_pairs,2)
        # multihot.shape == (n_pos_pairs,n_pred_cat)


        assert len(proposal_list) == len(label_list) 
        batch_size = len(proposal_list)
        pred_logits = [self.forward_propagation(proposal,pairid2trajids) for proposal,pairid2trajids in zip(proposal_list,pos_id_list)]

        # pred_logits: shape == (n_pos_pairs, n_pred_cat) 
        

        loss_dict = self.loss(pred_logits,label_list)

        total_loss = torch.stack(list(loss_dict.values())).sum()    # scalar tensor

        
        return total_loss, loss_dict


    def loss(self,pred_logits,label_list):
        
        pred_logits = torch.cat(pred_logits,dim=0) # (N_pos_pairs,n_pred_cat)
        labels = torch.cat(label_list)             # (N_pos_pairs,n_pred_cat)

        cls_loss = F.binary_cross_entropy_with_logits(pred_logits,labels,reduction='mean')
        # TODO use focal ?
        loss_dict = {
            "cls":cls_loss
        }
        return loss_dict



    def construct_triplet(self,proposal,pred_logits,pair_ids):
        # pred_cs.shape == (n_pred,2) center span
        
        pred_probs = torch.softmax(pred_logits,dim=-1)
        pred_scores,pred_catids = torch.topk(pred_probs,self.topk,dim=-1)  # shape == (n_anchors,k)
        pred_scores = pred_scores.reshape(-1) # shape == (n_ac*k,) # flatten as concatenate each row
        pred_catids = pred_catids.reshape(-1) # shape == (n_ac*k,)

        
        traj_duras = proposal.traj_durations.clone() # shape == (n_enti,2)
        n_traj = traj_duras.shape[0]

        enti_scores = proposal.scores            # shape == (n_enti,)
        enti_catids = proposal.cat_ids           # shape == (n_enti,)
        
        
        pred2so_ids = pair_ids          # enti index,  shape == (n_ac,2)
        pred2so_ids = torch.repeat_interleave(pred2so_ids,self.topk,dim=0) # shape == (n_ac*k,2)

        # filter the predicates linking to object/subject such that have no overlap
        dura_inters,dura_mask = dura_intersection_ts(traj_duras,traj_duras)  # shape == (n_traj,n_traj,2)
        dura_mask[range(n_traj),range(n_traj)] = 0
        pos_pred_mask = dura_mask[pred2so_ids[:,0],pred2so_ids[:,1]]  # shape = (n_ac*k,)
        if pos_pred_mask.sum() == 0:
            return None
        pos_pred_index = pos_pred_mask.nonzero(as_tuple=True)[0]

        pred2so_ids = pred2so_ids[pos_pred_index,:]  # shape == (n_pos,2)
        pred_scores =pred_scores[pos_pred_index]     # shape == (n_pos,)
        pred_catids =pred_catids[pos_pred_index]     # shape == (n_pos,)
        
        
        # triplets
        pred2so_catids = enti_catids[pred2so_ids] # shape == (n_pos,2)
        triplet_catids = torch.cat([pred_catids[:,None],pred2so_catids],dim=-1)  # shape == (n_pos,3) format: [pred_catid,subj_catid,obj_catid]
        
        # scores
        pred2so_scores = enti_scores[pred2so_ids]  # shape == (n_pos,2)
        triplet_scores = torch.cat([pred_scores[:,None],pred2so_scores],dim=-1) # shape == (n_pos,3)

        # filter the repeated triplets ( the post-processing in MM_paper1933)
        quintuples = torch.cat([triplet_catids,pred2so_ids],dim=-1)  # shape == (n_pos,5) format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
        try:
            uniq_quintuples,index_map = unique_with_idx_nd(quintuples)        # shape == (n_unique,5) format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
        except:
            print(quintuples.shape,pos_pred_mask.shape,pos_pred_mask.sum())
            print(dura_mask.sum(),dura_mask.shape)
            print(proposal.video_name)
            assert False

        uniq_triplet_ids = [idx[triplet_scores[idx,0].argmax()] for idx in index_map] # list of scalar tensor
        uniq_triplet_ids = torch.stack(uniq_triplet_ids)    # shape == (n_unique,)                          
        uniq_scores = triplet_scores[uniq_triplet_ids,:]      # shape == (n_unique,3)
                               
        uniq_dura_inters = dura_inters[uniq_quintuples[:,3],uniq_quintuples[:,4],:] # shape == (n_unique,2)
        #TODO sort by socre and select top100?


        # filter out triplets whose pred_cat is __background__
        mask = uniq_quintuples[:,0] != 0
        uniq_quintuples = uniq_quintuples[mask,:]
        uniq_scores = uniq_scores[mask,:]
        uniq_dura_inters = uniq_dura_inters[mask,:]
        

        if self.rt_triplets_topk > 0:
            # sort by score and select top200 (for save GPU memory when doing the grounding stage)
            top200ids = uniq_scores.mean(dim=-1).argsort(descending=True)[:self.rt_triplets_topk]
            uniq_scores = uniq_scores[top200ids,:]
            uniq_quintuples = uniq_quintuples[top200ids,:]
            uniq_dura_inters = uniq_dura_inters[top200ids,:]

        uniq_query_ids = torch.empty(size=(uniq_scores.shape[0],))

        ret = (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        )

        return ret

