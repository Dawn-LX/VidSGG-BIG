import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_func import dura_intersection_ts, tIoU,unique_with_idx_nd


def tiou_left_right(lr_1,lr_2):
    tiou = (torch.min(lr_1[:,1],lr_2[:,1]) + torch.min(lr_1[:,0],lr_2[:,0])) \
            / (torch.max(lr_1[:,1],lr_2[:,1]) + torch.max(lr_1[:,0],lr_2[:,0]))

    return tiou



def generalized_tIoU(duras1,duras2,broadcast=True):
    # gIoU = IoU - |C\(A U B)| / |C|  \in [-1,1]
    # one-dim IoU (tIoU) is just the above tIoU func without ``tiou[torch.logical_not(mask)] = 0``

    # duras1.shape == (n1,2)
    # duras2.shape == (n2,2)
    if broadcast:
        g_tiou = (torch.min(duras1[:,None,1],duras2[None,:,1]) - torch.max(duras1[:,None,0],duras2[None,:,0])) \
            / (torch.max(duras1[:,None,1],duras2[None,:,1]) - torch.min(duras1[:,None,0],duras2[None,:,0]))
    else:
        assert duras1.shape == duras2.shape
        g_tiou = (torch.min(duras1[:,1],duras2[:,1]) - torch.max(duras1[:,0],duras2[:,0])) \
            / (torch.max(duras1[:,1],duras2[:,1]) - torch.min(duras1[:,0],duras2[:,0]))


    return g_tiou   # shape == (n1,n2)


class DepthWiseSeparableConv1d(nn.Module):
    """
    Xception: Deep Learning with Depthwise Separable Convolutions 
    (https://arxiv.org/pdf/1610.02357.pdf)
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,bias=True):
        super().__init__()
        
        self.depth_wise = nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding=kernel_size//2,groups=in_channels,bias=bias)
        self.point_wise = nn.Conv1d(in_channels,out_channels,kernel_size=1,bias=bias)

        nn.init.kaiming_normal_(self.depth_wise.weight)
        nn.init.kaiming_normal_(self.point_wise.weight)
        if bias:
            nn.init.constant_(self.depth_wise.bias, 0.0)
            nn.init.constant_(self.point_wise.bias, 0.0)

    def forward(self,x):
        x1 = self.depth_wise(x)
        x2 = self.point_wise(x1)
        return x2

class PosEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        freqs = [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]
        phases = [0 if i % 2 == 0 else np.pi / 2 for i in range(d_model)]
        self.freqs = torch.Tensor(freqs)[:,None]  # (d_model,1)
        self.phases = torch.Tensor(phases)[:,None] # (d_model,1)
        self.d_model = d_model

    def forward(self, x):
        assert len(x.shape) == 3
        batch_size,d_model,length = x.shape
        assert d_model == self.d_model

        pos = torch.arange(length)[None,:].repeat(d_model, 1).float() # (d_model,length)

        pos_encoding = torch.sin(pos * self.freqs + self.phases)

        x = x + pos_encoding.to(x.device)[None,:,:]
        
        return x


class QANetEncoderLayer(nn.Module):
    """
    QANet: Combining local convolution with global self-attention for reading comprehension
    (https://arxiv.org/pdf/1804.09541.pdf)
    """
    def __init__(self,d_model,num_conv,kernel_size):
        super().__init__()

        self.convs = nn.ModuleList([DepthWiseSeparableConv1d(d_model, d_model, kernel_size) for _ in range(num_conv)])
        self.mh_attn = nn.MultiheadAttention(d_model,num_heads=8,dropout=0.1)

        self.fc = nn.Linear(d_model, d_model, bias=True)
        self.pos = PosEncoder(d_model)
        self.normb = nn.LayerNorm(d_model)
        self.norm_seq = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv)])
        self.norme = nn.LayerNorm(d_model)
        self.num_conv = num_conv
        self.dropout = 0.1
    
    def self_attn(self,x):
        # x.shape == (N,d_model,T)
        
        kqv = x.permute(2,0,1)  # (T,N,d_model)
        
        out = self.mh_attn(kqv,kqv,kqv)[0]  # (T,N,d_model)
        out = out.permute(1,2,0)            # (N,d_model,T)

        return out

    def forward(self, x):
        """
        input:  x (N, d_model, T)
        return:  (N,d_model,T)
        """
        # x (N, d_model, T)
        out = self.pos(x)
        res = out
        out = self.normb(out.transpose(1,2)).transpose(1,2)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.num_conv   # p_drop is the robability of an element to be zeroed (not kept)
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out       # shape == (N,d_model,T)
            out = self.norm_seq[i](out.transpose(1,2)).transpose(1,2)  # shape == (N,d_model,T)
        out = self.self_attn(out) # shape == (N,d_model,T)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        res = out                               # (N,d_model,T)
        out = self.norme(out.transpose(1, 2))   # (N,T,d_model)
        out = self.fc(out).transpose(1, 2)      # (N,d_model,T)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class DEBUG(nn.Module):
    """
    overlapped subj-obj feature + cat([subj_feat,obj_feat,pred_feat]) --> start/end
    """
    def __init__(self,config,is_train=True):
        super().__init__()
        self.is_train = is_train

        self.dim_feat = config["dim_feat"]
        self.dim_clsme = config["dim_clsme"]
        self.dim_hidden = config["dim_hidden"] # (128,)
        self.num_bins = config["num_bins"]
        
        self.loss_factor = config["loss_factor"]

        self.EntiNameEmb_path = config["EntiNameEmb_path"]
        self.PredNameEmb_path = config["PredNameEmb_path"]

        EntiNameEmb = np.load(self.EntiNameEmb_path)
        EntiNameEmb = torch.from_numpy(EntiNameEmb).float()
        self.EntiNameEmb = nn.Parameter(EntiNameEmb, requires_grad = True)  
        # shape == (num_enti_cats,dim_emb) == (81,300) # including background
        assert self.EntiNameEmb.shape == (81,300)

        PredNameEmb = np.load(self.PredNameEmb_path)
        PredNameEmb = torch.from_numpy(PredNameEmb).float()
        self.PredNameEmb = nn.Parameter(PredNameEmb, requires_grad = True)  
        # shape == (num_pred_cats,dim_emb) == (51,300) # including background
        assert self.PredNameEmb.shape == (51,300)
        self.pred_cats_all = torch.tensor(range(51))

        self.video_fc = nn.Linear(self.dim_feat,self.dim_hidden)
        self.query_fc = nn.Linear(self.dim_clsme,self.dim_hidden)
        self.temp_fc = nn.Linear(2,self.dim_hidden)
        self.vq_fc = nn.Linear(self.dim_hidden*4,self.dim_hidden)

        self.video_encoder = QANetEncoderLayer(self.dim_hidden,4,kernel_size = 7)
        self.query_encoder = QANetEncoderLayer(self.dim_hidden,4,kernel_size = 3)
        self.combined_encoder = QANetEncoderLayer(self.dim_hidden,4,kernel_size = 7)

        self.proj2sim = nn.Linear(self.dim_hidden,self.dim_hidden,bias=False)

        temp = nn.Sequential(
            DepthWiseSeparableConv1d(self.dim_hidden,self.dim_hidden,3),
            nn.ReLU()
        )
        temp2 = [copy.deepcopy(temp) for _ in range(4)] \
            + [DepthWiseSeparableConv1d(self.dim_hidden,self.num_bins,3)]
        temp3 = [copy.deepcopy(temp) for _ in range(4)] \
            + [DepthWiseSeparableConv1d(self.dim_hidden,2*self.num_bins,3),nn.Sigmoid()]
        
        self.cls_head = nn.Sequential(*temp2)
        self.conf_head = copy.deepcopy(self.cls_head)
        self.regr_head = nn.Sequential(*temp3)
        

    def forward(self,video_feature_list,data_list,score_th=0.5,tiou_th=0.5,bins_th=0.1,nms_th=0.5,with_gt_data=True):
        self.pred_cats_all = self.pred_cats_all.to(video_feature_list[0].device)
        if with_gt_data:
            # When evaluating the grounding stage only, take the gt_data as input and evaluate the grounding model itself only
            datas = [self.prepare_gt_data(gt) for gt in data_list]
            words_embs,inter_duras,targets,index_maps = list(zip(*datas))
        else:
            assert not self.is_train
            datas = [self.prepare_data(d) for d in data_list]
            words_embs,inter_duras = list(zip(*datas))


        if self.is_train:
            return self._forward_train(video_feature_list,words_embs,inter_duras,targets,index_maps)
        else:
            assert len(video_feature_list) == 1  # we set batch_size=1 at test time for simplicity
            self.bin_conf_th = bins_th
            self.score_th = score_th
            self.tiou_th = tiou_th
            self.nms_th = nms_th
            if (words_embs[0] is None) or (inter_duras[0] is None):
                return None,None
            

            pooled_se,bins_probs,bins_mask = self._forward_test_single(video_feature_list[0],words_embs[0],inter_duras[0])
            return pooled_se,bins_probs,bins_mask

    
    def get_gt_labels(self,target,n_clips):
        # target is normalized
        assert torch.all(target <= 1)
        clip_range = torch.linspace(0,1,n_clips,device=target.device)  # shape == (n_clips,)
        bins = torch.linspace(0,1,self.num_bins+1,device=target.device)  # 0~1 closed interval 
        target_ct = target.mean(dim=-1)
        
        offset = target_ct[:,None] - bins[None,:]  # (n_query,n_bins+1)
        bin_ids = (offset > 0).sum(dim=-1) - 1   # 0 ~ n_bins-1          # (n_query,)
        left =  clip_range[None,:] - target[:,0,None]  # shape == (n_query,n_clips)
        right = target[:,1,None] - clip_range[None,:]
        mask = (left <= 0) | (right <= 0)  # (n_query,n_clips)

        gt_ctness = torch.sqrt(torch.minimum(left,right) / torch.maximum(left,right))
        gt_ctness[mask] = 0         # (n_query,n_clips)  set `nan` as 0
        gt_scores = torch.ones_like(gt_ctness)
        gt_scores[mask] = 0
        gt_regrs = torch.stack([left,right],dim=-1)  # shape == (n_query,n_clips,2)


        ret  = (
            gt_regrs,          # shape == (n_query,n_clips,2)
            gt_ctness,         # shape == (n_query,n_clips)
            gt_scores,         # shape == (n_query,n_clips)
            bin_ids,         # shape == (n_query,)
        )
        return ret    

    
    def prepare_gt_data(self,gt_graph):
        if gt_graph.num_trajs==0 or gt_graph.num_preds==0:
            return None,None,None,None
        
        video_len = gt_graph.video_len
        # traj_bboxes = gt_graph.traj_bboxes  # list[tensor],each shape == (n_frames,4) # format: xyxy
        traj_bboxes = gt_graph.traj_bboxes  # list[tensor],each shape == (n_frames,4) # format: xyxy
        traj_cats = gt_graph.traj_cat_ids  # shape == (n_traj,)
        traj_duras = gt_graph.traj_durations  # shape == (n_traj,2)
        pred_durations = gt_graph.pred_durations  # shape == (n_pred,2)
        pred_cats  = gt_graph.pred_cat_ids     # shape == (n_pred,)
        pred2so_ids = torch.argmax(gt_graph.adj_matrix,dim=-1).t()  # enti index,  shape == (n_gt_pred,2)
        pred2so_cats = traj_cats[pred2so_ids] # shape == (n_pred,2)
        sub_dura = traj_duras[pred2so_ids[:,0],:]
        obj_dura = traj_duras[pred2so_ids[:,1],:]
        inter_dura, mask = dura_intersection_ts(sub_dura,obj_dura,broadcast=False) # (n_pred,2)

        query_tags = torch.cat(
            [pred_cats[:,None],pred2so_cats,inter_dura],dim=-1
        ) # (n_pred,5)  format: [pred_catid,subj_catid,obj_catid,so_s,so_e]
        unique_tags,index_map = unique_with_idx_nd(query_tags)

        pred_emb = self.PredNameEmb[unique_tags[:,0],:] # shape == (n_uniq,dim_meb) == (n_pr,300)
        sub_emb  = self.EntiNameEmb[unique_tags[:,1],:] # shape == (n_uniq,dim_emb) 
        obj_emb  = self.EntiNameEmb[unique_tags[:,2],:] # shape == (n_uniq,dim_emb)
        words_emb = torch.stack([sub_emb,pred_emb,obj_emb],dim=1)  # shape == (n_uniq,3,dim_emb)
        temporal_info = unique_tags[:,3:].float() / video_len  # (n_uniq,2)
        
        target = pred_durations.float() / video_len

        if self.is_train:
            ## construct negative samples
            uniq_so_tags,index_map_so = unique_with_idx_nd(unique_tags[:,1:])
            rand_pred_cats = []
            for im in index_map_so:
                # print(query_tags[im,0],im)
                device = query_tags.device
                mask = torch.zeros(size=(51,),dtype=torch.bool,device=device).scatter_(0,query_tags[im,0],1)
                other_cats = self.pred_cats_all[~mask]
                rand_ids = torch.randperm(len(other_cats),device=device)
                selected = other_cats[rand_ids][:len(im)]
                rand_pred_cats.append(selected)
            rand_pred_cats= torch.cat(rand_pred_cats,dim=0)
            # xx = torch.cat(index_map_so)
            # print(len(rand_pred_cats),words_emb.shape,xx.shape)
            neg_pred_emb = self.PredNameEmb[rand_pred_cats,:]  # (n_uniq,dim_emb)
            neg_words_emb = torch.stack([sub_emb,neg_pred_emb,obj_emb],dim=1)  # shape == (n_neg,3,dim_emb)
            neg_temporal_info = temporal_info.clone()

            words_emb = torch.cat([words_emb,neg_words_emb],dim=0)
            temporal_info = torch.cat([temporal_info,neg_temporal_info],dim=0)
        

        return words_emb,temporal_info,target,index_map


    
    def prepare_data(self,datas):

        uniq_quintuples,uniq_dura_inters,video_len = datas

        # model_0v4  returns:
        # uniq_quintuples,    # shape == (n_unique,5)  format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
        # uniq_scores,        # shape == (n_unique,)
        # uniq_pred_confs     # shape == (n_unique,)
        # uniq_dura_inters    # shape == (n_unique,2) 

        pred_emb = self.PredNameEmb[uniq_quintuples[:,0],:] # shape == (n_uniq,dim_meb) == (n_pr,300)
        sub_emb  = self.EntiNameEmb[uniq_quintuples[:,1],:] # shape == (n_uniq,dim_emb) 
        obj_emb  = self.EntiNameEmb[uniq_quintuples[:,2],:] # shape == (n_uniq,dim_emb)
        words_emb = torch.stack([sub_emb,pred_emb,obj_emb],dim=1)  # shape == (n_uniq,3,dim_emb)
        temporal_info = uniq_dura_inters.float() / video_len  # (n_uniq,2)
        
        

        return words_emb,temporal_info
     

    def forward_propagation(self,video_feature,words_emb,inter_dura):
        # video_feature: (T,dim_video)   T = n_clips
        # words_emb:     (n_query,L,dim_emb), L = 3
        n_clips = video_feature.shape[0]

        video_emb = self.video_fc(video_feature).t()[None,:,:]  # (1,dim_hidden,T)
        words_emb = self.query_fc(words_emb).permute(0,2,1)     # (n_query,dim_hidden,L)
        temporal_emb = self.temp_fc(inter_dura)                 # (n_query,dim_hidden)
        query_emb = words_emb + temporal_emb[:,:,None]

        video_emb = self.video_encoder(video_emb)  # (1,dim_hidden,T)
        query_emb = self.query_encoder(query_emb)  # (n_query,dim_hidden,L)
        
        # shape (T, n_query)
        n_query = query_emb.shape[0]
        sim_matrix = torch.matmul(
            self.proj2sim(video_emb.transpose(1,2)).expand(n_query,-1,-1),  # (n_query,T,dim_hidden)
            query_emb
        )  # (n_query,T,L)  n_query as batch_size, because each query is served as one language query in DEBUG
        
        sim_matrix_r = torch.softmax(sim_matrix,dim=2)  # (n_query,T,L)
        sim_matrix_c =  torch.softmax(sim_matrix,dim=1) # (n_query,T,L)
        sim_matrix_rc = torch.matmul(sim_matrix_r,sim_matrix_c.transpose(1,2))  # shape == (n_query,T,T)

        video_emb = video_emb.expand(n_query,-1,-1).transpose(1,2)  # (n_query,T,dim_hidden)
        mat_A = torch.matmul(sim_matrix_r,query_emb.transpose(1,2))  # batched matmul, shape==(n_query, T, dim_hidden)
        mat_B = torch.matmul(sim_matrix_rc,video_emb) # batched matmul, shape==(n_query, T, dim_hidden)


        combined_feature = torch.cat([
            video_emb,
            mat_A,
            mat_A*video_emb,
            mat_B*video_emb
        ],dim=-1)  # shape == (n_query, T, dim_hidden*4)

        combined_feature = self.vq_fc(combined_feature).transpose(1,2)  # (n_query,dim_hidden,T)
        combined_feature = self.combined_encoder(combined_feature)      # (n_query,dim_hidden,T)

        regrs = self.regr_head(combined_feature).transpose(1,2)                   # (n_query,T,2*k)
        conf_logits = self.conf_head(combined_feature).transpose(1,2)             # (n_query,T,k)
        cls_logits = self.cls_head(combined_feature).transpose(1,2)               # (n_query,T,k)
        return regrs,conf_logits,cls_logits

    def _forward_train(self,video_feature_list,words_embs,inter_duras,targets,index_maps):
        batch_size = len(video_feature_list)
        results = [self.forward_propagation(video_feature,words_emb,inter_dura) for video_feature,words_emb,inter_dura in zip(video_feature_list,words_embs,inter_duras)]
        regrs,conf_logits,cls_logits = list(zip(*results))
        n_clips = [r.shape[1] for r in regrs]

        labels = [self.get_gt_labels(tgt,n) for tgt,n in zip(targets,n_clips)]

        bin_ids = [label[-1] for label in labels]
        mapped_predictions = [self.map2bins(re,co,cl,sl,im) for re,co,cl,sl,im in zip(regrs,conf_logits,cls_logits,bin_ids,index_maps)]

        loss_dict = self.loss(mapped_predictions,labels,index_maps)

        total_loss = torch.stack(list(loss_dict.values())).sum()    # scalar tensor        
        return total_loss, loss_dict
    
    def map2bins(self,regrs,conf_logits,cls_logits,bin_ids,index_map):
        # regrs         (n_uniq*2,n_clips,2*n_bins)
        # conf_logits   (n_uniq*2,n_clips,n_bins)
        # cls_logits    (n_uniq*2,n_clips,n_bins)
        n_uniq2,n_clips,_ = regrs.shape
        # print(regrs.shape)
        n_uniq = n_uniq2//2
        regrs = regrs.reshape(n_uniq2,n_clips,2,self.num_bins)

        pos_conf = []
        neg_conf = []
        pos_cls = []
        neg_cls = []
        pos_regrs = []
        for i,imp in enumerate(index_map):     # loop for n_uniq
            bins = bin_ids[imp]  # (n_dup_i,)
            bins_mask = torch.zeros(size=(self.num_bins,),dtype=torch.bool,device=bins.device).scatter_(0,bins,1)

            pos_conf_i = conf_logits[i,:,bins]          # (n_clips,n_dup_i)
            neg_conf_i = conf_logits[i,:,~bins_mask]    # (n_clips,n_neg_i) n_neg_i: number of negative bins in this uniq_i 
            pos_cls_i = cls_logits[i,:,bins]
            neg_cls_i = cls_logits[i,:,~bins_mask]  
            regr_i = regrs[i,:,:,bins]  # (n_clips,2,n_dup_i) 

            pos_conf.append(pos_conf_i)
            neg_conf.append(neg_conf_i)
            pos_cls.append(pos_cls_i)
            neg_cls.append(neg_cls_i)
            pos_regrs.append(regr_i)
        

        pos_conf = torch.cat(pos_conf,dim=-1).t()   # (n_clips,n_query)  --> (n_query,n_clips)
        neg_conf = torch.cat(neg_conf,dim=-1).t()   # (n_clips,n_neg) --> (n_neg,n_clips)  n_negs: number of negative bins in all n_uniq
        pos_cls = torch.cat(pos_cls,dim=-1).t() 
        neg_cls = torch.cat(neg_cls,dim=-1).t()
        pos_regrs = torch.cat(pos_regrs,dim=-1).permute(2,0,1)  # (n_clips,2,n_query) --> (n_query,n_clips,2)
        
        # for negativate samples:
        neg_conf2 = conf_logits[n_uniq:,:,:].permute(0,2,1).reshape(n_uniq*self.num_bins,n_clips)
        neg_cls2 = cls_logits[n_uniq:,:,:].permute(0,2,1).reshape(n_uniq*self.num_bins,n_clips)

        neg_conf = torch.cat([neg_conf,neg_conf2],dim=0)
        neg_cls = torch.cat([neg_cls,neg_cls2],dim=0)

        ret = (
            pos_conf,           # (n_query,n_clips)
            neg_conf,           # as above
            pos_cls,            # as above
            neg_cls,            # as above
            pos_regrs,          # (n_query,n_clips,2)
        )
        return ret

    def loss(self,mapped_predictions,labels,index_maps):
        batch_size = len(labels)

        pos_confs = []
        neg_confs = []
        pos_clss = []
        neg_clss = []
        pos_regrs = []
        gt_regrs = []
        gt_ctness = []
        gt_scores = []
        for pred,label,index_map in zip(mapped_predictions,labels,index_maps):
            pos_conf,neg_conf,pos_cls,neg_cls,pos_regr  = pred
            gt_regr,gt_ctnes,gt_score,bins_id = label
            index_map = torch.cat(index_map)  # (n_query,)
            gt_regr = gt_regr[index_map,:,:]
            gt_ctnes = gt_ctnes[index_map,:]
            gt_score = gt_score[index_map,:]
            # bins_id has been applied with `index_map` in self.map2bins

            # pos_conf,           # (n_query,n_clips)
            # neg_conf,           # as above
            # pos_cls,            # as above
            # neg_cls,            # as above
            # pos_regr,           # (n_query,n_clips,2)
            # bins_conf_target   # (n_uniq,n_bins)
            # bins_logit         # (n_uniq,n_bins)
            # gt_regr             (n_query,n_clips,2)
            # gt_ctnes             (n_query,n_clips)
            # gt_score             (n_query,n_clips)
            # bins_id             (n_query,)

            pos_confs.append(pos_conf.reshape(-1))
            neg_confs.append(neg_conf.reshape(-1))
            pos_clss.append(pos_cls.reshape(-1))
            neg_clss.append(neg_cls.reshape(-1))
            pos_regrs.append(pos_regr.reshape(-1,2))

            gt_regrs.append(gt_regr.reshape(-1,2))
            gt_ctness.append(gt_ctnes.reshape(-1))
            gt_scores.append(gt_score.reshape(-1))


        pos_confs = torch.cat(pos_confs,dim=0)  # (N_query_N_clips,)
        neg_confs = torch.cat(neg_confs,dim=0)
        gt_ctness = torch.cat(gt_ctness,dim=0)

        pos_clss = torch.cat(pos_clss,dim=0)    # (N_query_N_clips,)
        neg_clss = torch.cat(neg_clss,dim=0)
        gt_scores = torch.cat(gt_scores,dim=0)

        pos_regrs = torch.cat(pos_regrs,dim=0)  # (N_query_N_clips,2)
        gt_regrs = torch.cat(gt_regrs,dim=0)

        neg_target = torch.zeros_like(neg_confs)
        mask = gt_ctness > 0
        
        pos_cls_loss = F.binary_cross_entropy_with_logits(pos_clss,gt_scores,reduction='mean')
        neg_cls_loss = F.binary_cross_entropy_with_logits(neg_clss,neg_target,reduction='mean')
        neg_ct_loss = F.binary_cross_entropy_with_logits(neg_confs,neg_target,reduction='mean')
        
        if mask.sum() == 0:
            pos_ct_loss = torch.zeros_like(pos_cls_loss)
            regr_loss = torch.zeros_like(pos_cls_loss)
        else:
            pos_ct_loss = F.binary_cross_entropy_with_logits(pos_confs[mask],gt_ctness[mask],reduction='mean')
            regr_loss = tiou_left_right(pos_regrs[mask,:],gt_regrs[mask,:])
            regr_loss = (-1 * (regr_loss+1e-6).log()).mean()
            

        pos_cls_loss *= self.loss_factor["classification"]
        neg_cls_loss *= self.loss_factor["classification"]
        pos_ct_loss *= self.loss_factor["centerness"]
        neg_ct_loss *= self.loss_factor["centerness"]
        regr_loss *= self.loss_factor["regression"]
        
        loss_dict = {
            "pos_cls":pos_cls_loss,
            "neg_cls":neg_cls_loss,
            "pos_ct":pos_ct_loss,
            "neg_ct":neg_ct_loss,
            "regr":regr_loss
        }
        return loss_dict


    def _forward_test_single(self,video_feature,words_emb,inter_dura):
        # inter_dura is normalized
        regrs,conf_logits,cls_logits = self.forward_propagation(video_feature,words_emb,inter_dura)
        confs = conf_logits.sigmoid()
        fg_probs = cls_logits.sigmoid()
        scores = confs * fg_probs       # (n_uniq, n_clips, k)

        bins_probs = torch.max(scores,dim=1)[0]  # (n_uniq,k)
        bins_probs = torch.constant_pad_nd(bins_probs,pad=(0,1),value=1.0)  # (n_uniq,k+1)
        bins_mask = bins_probs > self.bin_conf_th # (n_uniq,k+1)
        
        
        pooled_se = self.temporal_pooling(regrs,scores) # (n_uniq,k,2)
        
        # inter_dura         (n_uniq,2)
        overlap_mask = []
        for k in range(self.num_bins):
            pooled_se_k = pooled_se[:,k,:]  # (n_uniq,2)
            se_spo,mask = dura_intersection_ts(inter_dura,pooled_se_k,broadcast=False)
            pooled_se[:,k,:] = inter_dura.clone()
            pooled_se[mask,k,:] = se_spo[mask,:]
            overlap_mask.append(mask)  # (n_uniq,)
        overlap_mask = torch.stack(overlap_mask,dim=-1)  # (n_uniq,k)
        overlap_mask = torch.constant_pad_nd(overlap_mask,pad=(0,1),value=1)  # (n_uniq,k+1)
        
        pooled_se = torch.cat([pooled_se,inter_dura[:,None,:]],dim=1)  # (n_uniq,k+1,2)

        bins_mask_nms = self.temporal_nms(pooled_se,bins_probs)
        bins_mask = bins_mask & overlap_mask & bins_mask_nms         # (n_uniq, k+1)

        #--------------- make sure each row of bins_mask has at least one `True`
        allFalse_rowids = (bins_mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
        if allFalse_rowids.numel()>0:
            max_col_ids = bins_probs[allFalse_rowids,:].max(dim=-1)[1]
            bins_mask[allFalse_rowids,max_col_ids] = 1
        # ----------------


        #### improve  
        # for thoses with small  max_bins_prob, they  might be false positives returned by the classification stage
        # i.e., the grounding stage can correct the classification stage to some extent.
        mask = bins_probs[:,:-1].max(-1)[0] <= self.bin_conf_th
        bins_probs[mask,-1] = 0.0   # set the score of `subj-obj overlap` as 0.0
        #### 


        return pooled_se,bins_probs,bins_mask

    def eval_tiou(self,prediction_se,bins_mask,target,index_map):
        n_uniq = prediction_se.shape[0]
        if bins_mask is None:  # for baseline
            assert len(prediction_se.shape) == 2
            # prediction_se.shape == (n_uniq,2)
            inter_dura = prediction_se
            bins_mask = torch.ones(size=(n_uniq,self.num_bins),device=inter_dura.device,dtype=torch.bool)
            pooled_cl = torch.rand(size=(n_uniq,self.num_bins,2),device=inter_dura.device)
            s = pooled_cl[:,:,1] - pooled_cl[:,:,0]/2
            e = pooled_cl[:,:,1] + pooled_cl[:,:,0]/2
            pooled_se = torch.stack([s,e],dim=-1)  # (n_uniq,num_bins,2)
            # print(pooled_se.shape,"pooled_se.shape")
            for k in range(self.num_bins):
                pooled_se_k = pooled_se[:,k,:]  # (n_uniq,2)
                # print(pooled_se_k)
                se_spo,mask = dura_intersection_ts(inter_dura,pooled_se_k,broadcast=False)
                pooled_se[:,k,:] = inter_dura.clone()
                pooled_se[mask,k,:] = se_spo[mask,:]
            prediction_se = pooled_se
        else:
            assert len(prediction_se.shape) == 3
            # prediction_se.shape == (n_uniq,n_bins,2)  for model prediction

        tiou_all = []
        for i,im in enumerate(index_map):
            mask = bins_mask[i,:]  # (n_bins,)
            dup_tgt = target[im,:]  # (n_dup,2)
            se = prediction_se[i,mask,:]   # (n_pos,2)

            tiou_matrix = tIoU(dup_tgt,se,broadcast=True)  # (n_dup,n_pos)
            tiou = tiou_matrix.max(dim=-1)[0]  # (n_dup,)   # 这里引入了先验， 没法取score最高的前dup个，因为不知道谁对应谁
            tiou_all.append(tiou)
        
        tiou_all = torch.cat(tiou_all)  # (n_query,)
        return tiou_all

    
    def eval_f1score(self,prediction_se,bins_mask,target,index_map,tiou_ths=[0.5]):
        n_uniq = prediction_se.shape[0]
        if bins_mask is None:  # for baseline
            assert len(prediction_se.shape) == 2
            # prediction_se.shape == (n_uniq,2)
            inter_dura = prediction_se
            bins_mask = torch.ones(size=(n_uniq,self.num_bins),device=inter_dura.device,dtype=torch.bool)
            pooled_cl = torch.rand(size=(n_uniq,self.num_bins,2),device=inter_dura.device)
            s = pooled_cl[:,:,1] - pooled_cl[:,:,0]/2
            e = pooled_cl[:,:,1] + pooled_cl[:,:,0]/2
            pooled_se = torch.stack([s,e],dim=-1)  # (n_uniq,num_bins,2)
            # print(pooled_se.shape,"pooled_se.shape")
            for k in range(self.num_bins):
                pooled_se_k = pooled_se[:,k,:]  # (n_uniq,2)
                # print(pooled_se_k)
                se_spo,mask = dura_intersection_ts(inter_dura,pooled_se_k,broadcast=False)
                pooled_se[:,k,:] = inter_dura.clone()
                pooled_se[mask,k,:] = se_spo[mask,:]
            prediction_se = pooled_se
        else:
            assert len(prediction_se.shape) == 3
            # prediction_se.shape == (n_uniq,n_bins,2)  for model prediction

        n_hits = {k:[] for k in tiou_ths}
        n_tgts = []
        n_predictions = []
        for i,im in enumerate(index_map):
            mask = bins_mask[i,:]  # (n_bins,)
            dup_tgt = target[im,:]  # (n_dup,2)
            se = prediction_se[i,mask,:]   # (n_pos,2) n_pos: 1 ~ n_bins
            tiou_matrix = tIoU(dup_tgt,se,broadcast=True)  # (n_dup,n_pos)
            n_tgts.append(dup_tgt.shape[0])
            n_predictions.append(se.shape[0])
            for tiou_th in tiou_ths:
                hit_matrix = tiou_matrix > tiou_th
                is_hit = hit_matrix.sum(dim=-1) > 0  # (n_dup,)
                n_hits[tiou_th].append(is_hit.sum().item())
        
        recalls = {}
        precisions = {}
        f1scores = {}
        total_tgts = np.sum(n_tgts)
        total_predictions = np.sum(n_predictions)
        for tiou_th in tiou_ths:
            n_hits_i = np.sum(n_hits[tiou_th])
            recall_i = n_hits_i / total_tgts
            precision_i = n_hits_i / total_predictions
            recalls[tiou_th] = recall_i
            precisions[tiou_th] = precision_i
            f1scores[tiou_th] = 2*precision_i*recall_i / (precision_i + recall_i + 1e-6)
        return recalls,precisions,f1scores
    
    def _nms(self,boxes1d,probs,nms_th):

        index = probs.argsort()     # sorted_probs = probs[index]  # ascending (small -> large)
        # inv_index = index.argsort() # probs = sorted_probs[inv_index]
        tiou_matrix = tIoU(boxes1d,boxes1d)
        kept_ids = []
        while index.numel()>0:
            idx = index[-1]
            kept_ids.append(idx)
            left_ids = (tiou_matrix[idx,index[:-1]] < nms_th).nonzero(as_tuple=True)[0]
            index = index[left_ids]
        
        kept_ids = torch.stack(kept_ids,dim=0) # (n_left)
        kept_boxes1d = boxes1d[kept_ids,:]     # (n_left,2)
        return kept_boxes1d,kept_ids

    def temporal_nms(self,prediction_se,bins_probs):
        # prediction_se.shape == (n_uniq,n_bins,2) 
        # bins_probs.shape == (n_uniq,n_bins)
        n_uniq,n_bins = bins_probs.shape
        bins_mask = []
        for i in range(n_uniq):
            _,kept_ids = self._nms(prediction_se[i,:,:],bins_probs[i,:],self.nms_th)  # (n_left,)  n_left=1~n_bins
            mask = torch.zeros(size=(n_bins,),device=bins_probs.device,dtype=torch.bool)
            mask = mask.scatter_(0,kept_ids,1)
            bins_mask.append(mask)
        
        bins_mask = torch.stack(bins_mask,dim=0)
        return bins_mask

    def temporal_pooling(self,regrs,scores):
        # regrs         (n_uniq, n_clips, 2*k)
        # confs         (n_uniq, n_clips, k), k==self.num_bins
        # fg_probs      (n_uniq, n_clips, k)
        
        n_uniq,n_clips,_ = scores.shape
        regrs = regrs.reshape(n_uniq,n_clips,2,self.num_bins)

        clip_range = torch.linspace(0,1,n_clips,device=regrs.device)  # shape == (n_clips,) (n_uniq,n_clips,2,k)
        start = clip_range[None,:,None] - regrs[:,:,0,:]    # (n_uniq, n_clips, k)
        end = clip_range[None,:,None] + regrs[:,:,1,:]      # (n_uniq, n_clips, k)
        duras = torch.stack([start,end],dim=-1)             # (n_uniq, n_clips, k,2)

        pooled_se = []
        for qid in range(n_uniq):
            pooled_se_q = []
            for k in range(self.num_bins):
                score = scores[qid,:,k]  # (n_clips,)
                top_score,top_score_id = torch.max(score,dim=0)
                mask1 = score > self.score_th * top_score  # (n_clips,)
                dura = duras[qid,:,k,:]  # (n_clips,2)
                tiou_mat = generalized_tIoU(dura,dura) # (n_clips,n_clips)
                tiou_mask = tiou_mat > self.tiou_th
                row_ids,col_ids = tiou_mask.nonzero(as_tuple=True)
                select_ids = col_ids[row_ids == top_score_id]
                mask2 = torch.zeros_like(mask1).scatter_(0,select_ids,1)
                mask  = mask1 & mask2
                # print(mask.shape,mask.sum(),"mask--------")
                dura = dura[mask,:]  # (n_pos,2)
                start = torch.min(dura[:,0],dim=0)[0]
                end = torch.max(dura[:,1],dim=0)[0]
                pooled_se_q.append(
                    torch.stack([start,end])  # (2,)
                )
            pooled_se_q = torch.stack(pooled_se_q,dim=0)  # (k,2)
            pooled_se.append(pooled_se_q)
        pooled_se = torch.stack(pooled_se,dim=0) # shape == (n_uniq,k,2)

        

        return pooled_se

if __name__ == "__main__":


    config = dict(
        clip_len = 16,
        clip_step = 8,
        dim_feat = 1024,
        dim_clsme = 300,
        dim_hidden = 128,
        num_bins  = 10,
        EntiNameEmb_path = "prepared_data/vidor_EntiNameEmb.npy",
        PredNameEmb_path = "prepared_data/vidor_PredNameEmb.npy",
        loss_factor = dict(
            classification = 1.0,
            confidence = 1.0,
            regression = 1.0
        )
    )
    model = DEBUG(config)

    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(total_num,trainable_num)


