import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor


import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.utils_func import vIoU_ts,dura_intersection_ts,unique_with_idx_nd,stack_with_padding
from utils.categories_v2 import vidvrd_CatId2name,vidvrd_PredId2name,vidor_CatId2name,vidor_PredId2name

## notations: 
# dim_q == dim_query == dim_pred
# dim_e == dim_enti
# n_e == n_enti == n_traj == n_trajs
# n_q === n_pred == n_querys == n_querys

def stack_with_repeat_2d(tensor_list,dim):
    assert len(tensor_list[0].shape) == 2
    device = tensor_list[0].device
    shape_list = [t.shape for t in tensor_list]
    num_rows = torch.tensor([sp[0] for sp in shape_list])
    num_cols = torch.tensor([sp[1] for sp in shape_list])
    # assert num_rows[0]
    if torch.all(num_rows == num_rows[0]):
        max_L = num_cols.max()
        repeat_dim=1
    elif torch.all(num_cols == num_cols[0]):
        max_L = num_rows.max()
        repeat_dim=0
    else:
        assert False
    
    after_repeat = []
    for tensor in tensor_list:
        L = tensor.shape[repeat_dim]
        n_pad = L - (max_L % L)
        ones = [1]*max_L
        zeros = [0]*n_pad
        total = torch.tensor(ones + zeros,device=device)
        total = total.reshape(-1,L)
        repeats_ = total.sum(dim=0)
        after_repeat.append(
            tensor.repeat_interleave(repeats_,dim=repeat_dim)
        )
    return torch.stack(after_repeat,dim=dim)


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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.normalize_before = normalize_before

    def _get_activation_fn(self,activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # with positional embedding
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # q == k
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # print(src.shape,"src.shape")
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class RoleAttnDecoderLayer(nn.Module):

    def __init__(self, dim_pred, nhead, n_preds,dim_enti,dim_att,dim_ffn,dropout=0.1):
        super().__init__()
        self.dim_pred = dim_pred
        self.dim_enti = dim_enti
        self.num_querys = n_preds
        self.dim_ffn = dim_ffn
        self.dim_att = dim_att
        self.self_attn = nn.MultiheadAttention(dim_pred, nhead, dropout=dropout)
        
        fc_rolewise = nn.Sequential(
            nn.Linear(self.dim_enti,self.dim_pred),
            nn.ReLU(),
            nn.Linear(self.dim_pred,self.dim_pred)
        )
        self.fc_rolewise = _get_clones(fc_rolewise, 2)

        self.fc_enti2att = nn.Linear(self.dim_enti,self.dim_att)
        self.fc_pred2att = nn.Linear(self.dim_pred,self.dim_att)

        # Implementation of Feedforward model
        self.fc2 = nn.Sequential(
            nn.Linear(dim_pred, dim_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, dim_pred)
        )

        self.norm1 = nn.LayerNorm(dim_pred)
        self.norm2 = nn.LayerNorm(dim_pred)
        self.norm3 = nn.LayerNorm(dim_pred)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,pred_query,pos_emb,enco_output):
        
        ## self-attention
        v = pred_query[:,None,:]
        q = k = self.with_pos_embed(v, pos_emb[:,None,:])
        pred_query2 = self.self_attn(k, q, v)[0].squeeze(1)  # shape == (n_querys, dim_pred)
        
        ## add&norm_1
        pred_query = self.norm1(pred_query + pred_query2)

        ## cross-attention
        pred_query = self.with_pos_embed(pred_query, pos_emb)
        enti2att = self.fc_enti2att(enco_output)    # shape == (n_enti, dim_pred)
        pred2att = self.fc_pred2att(pred_query)  # shape == (n_querys, dim_pred)

        enti2att_subjobj = (enti2att[:,:self.dim_att//2],enti2att[:,self.dim_att//2:])  # NOTE this operation is equivalent to using self.fc_enti2att_s and self.fc_enti2att_o with half dim_att
        pred2att_subjobj = (pred2att[:,:self.dim_att//2],pred2att[:,self.dim_att//2:])  

        att_matrx = []
        for i in range(2):
            enti2att = enti2att_subjobj[i].t()
            pred2att = pred2att_subjobj[i]
            
            att_mat_i = torch.matmul(pred2att,enti2att) / np.sqrt(self.dim_enti)
            att_matrx.append(att_mat_i)
        
        att_matrx = torch.stack(att_matrx,dim=0)  # shape == (2,n_querys,n_trajs) 
        att_matrx_enti = torch.softmax(att_matrx,dim=2)
        att_matrx_role = torch.softmax(att_matrx,dim=0)
        att_matrx = att_matrx_enti * att_matrx_role  # shape == (2,n_querys,n_trajs) 
        
        role_queries = []
        for idx, fc in enumerate(self.fc_rolewise):
            values = torch.matmul(att_matrx[idx,:,:],enco_output)  # shape == (n_querys,dim_enti)
            role_q = fc(values)  # shape == (n_querys,dim_pred)
            role_queries.append(role_q)
        role_queries = torch.stack(role_queries,dim=0).sum(dim=0)  # shape == (n_querys,dim_pred)

        ## add&norm_2
        pred_query = self.norm2(pred_query + role_queries)

        ## feedforward
        pred_query2 = self.fc2(pred_query)
        
        ## add&norm_3
        pred_query = self.norm3(pred_query + pred_query2)
        
        return pred_query,att_matrx


def SinePosEmb(length,d_model):

    freqs = [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]
    phases = [0 if i % 2 == 0 else np.pi / 2 for i in range(d_model)]
    freqs = torch.Tensor(freqs)[None,:]  # (1, d_model)
    phases = torch.Tensor(phases)[None,:] # (1, d_model)
    pos = torch.arange(length)[:,None].repeat(1,d_model).float() # (length,d_model)
    pos_encoding = torch.sin(pos * freqs + phases)

    return pos_encoding

class BIG_C(nn.Module): 

    def __init__(self,config,is_train=True):
        super(BIG_C, self).__init__()
        self.is_train = is_train

        ## 1. configs
        # 1.1 model configs
        self.num_pred_cats = config["num_pred_cats"]
        self.num_enti_cats = config["num_enti_cats"]
        self.dim_feat = config["dim_feat"]          # 2048 or 1024 (dimension of each bbox's RoI feature, depend on the detector)
        self.dim_clsme = config["dim_clsme"]
        self.dim_enti = config["dim_enti"]
        self.dim_pred = config["dim_pred"]
        self.dim_att  = config["dim_att"] 
        self.dim_ffn  = config["dim_ffn"]
        
        self.enco_pool_len = config["enco_pool_len"]
        self.n_enco_layers = config["n_enco_layers"]
        self.n_deco_layers = config["n_deco_layers"]
        self.n_att_head = config["n_att_head"]
        self.num_querys = config["num_querys"]
        self.num_anchors = self.num_querys   # NOTE this `num_anchors` has been deprecated, this term is used in some ancient code

        self.bias_matrix_path = config["bias_matrix_path"]
        self.EntiNameEmb_path = config.get("EntiNameEmb_path",None)
        self.use_clsme = config["use_clsme"]

        if self.EntiNameEmb_path is None:  # for trajs that have classseme feature
            self.EntiNameEmb = None
        else:                              # otherwise, use category of traj to get a classeme feature
            EntiNameEmb = np.load(self.EntiNameEmb_path)
            EntiNameEmb = torch.from_numpy(EntiNameEmb).float()
            self.EntiNameEmb = nn.Parameter(EntiNameEmb, requires_grad = False)  
            # shape == (num_enti_cats,dim_emb) == (81,300) or (36, 300) # including background
            assert self.EntiNameEmb.shape == (self.num_enti_cats,self.dim_clsme)
        
        # 1.2 training configs
        self.neg_weight = config["neg_weight"]
        self.loss_factor = config["loss_coeff_dict"]    # loss coefficient dictionary
        self.cost_factor = config["cost_coeff_dict"]
        self.positive_vIoU_th = config["positive_vIoU_th"]

        
        ## 2. queries & anchors & frequency_bias
        pos_embedding = SinePosEmb(self.num_querys,self.dim_pred)
        self.pos_embedding = nn.Parameter(pos_embedding,requires_grad=False)
        self.pred_query_init = nn.Parameter(torch.FloatTensor(self.num_querys,self.dim_pred),requires_grad=True)
        
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

        ## 4. layers for encoder & decoder
        encoder_layer = TransformerEncoderLayer(
            self.dim_enti, self.n_att_head, self.dim_ffn,
            dropout=0.1, activation='relu', normalize_before=False
        )
        self.encoder_layers = _get_clones(encoder_layer, self.n_enco_layers)

        decoder_layer = RoleAttnDecoderLayer(
            self.dim_pred,self.n_att_head,self.num_querys,
            self.dim_enti,self.dim_att,self.dim_ffn,dropout=0.1
        )
        self.decoder_layers = _get_clones(decoder_layer, self.n_deco_layers)
        
        ## 5. layers for classification
        if self.use_clsme:
            dim_pred_query = self.dim_pred + self.dim_clsme*2 + self.dim_enti*2
        else:
            dim_pred_query = self.dim_pred + self.dim_enti*2
        
        self.fc_pred2logits = nn.Sequential(
            nn.Linear(dim_pred_query, self.dim_ffn),
            nn.ReLU(),
            nn.Linear(self.dim_ffn,self.num_pred_cats)
        )


        self._reset_parameters()

    
    def _reset_parameters(self):
        skip_init_param_names = [
            "bias_matrix",
            "EntiNameEmb",
            "pred_query_init"
        ]
        for name,p in self.named_parameters():
            if name in skip_init_param_names:  
                print("skip init param: {}".format(name))
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.normal_(self.pred_query_init,mean=0,std=0.1)
    
    
    def forward(self,proposal_list,gt_graph_list=None,topk=3):

        if self.is_train:
            assert gt_graph_list != None
            return self._forward_train(proposal_list,gt_graph_list)
        else: 
            self.topk = topk
            return self._forward_test(proposal_list)


    def _forward_test(self,proposal_list):

        triplets = []
        pred_queries = []
        pred_centerspan = []
        pred_regression = []
        xx = []
        for ii,proposal in enumerate(proposal_list):
            if proposal.num_proposals == 0:  # train 的时候 num_proposal == 0 的会被过滤掉
                triplets.append(None)
                pred_queries.append(None)
                pred_regression.append(None)
            else:
                pred_query,pred_logits,att_matrx = self.encode2decode(proposal)

                ret = self.construct_triplet(proposal,pred_logits,att_matrx)
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



    def encode2decode(self,proposal):
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
        output = enti2enco[:,None,:]
        for layer in self.encoder_layers:
            output = layer(output)
        enco_output = output.squeeze(1)  # shape == (n_enti,dim_state)

        ## decode 
        pred_queries = self.pred_query_init
        for layer in self.decoder_layers:
            pred_queries,att_matrx = layer(pred_queries,self.pos_embedding,enco_output)
        
        if self.use_clsme and (self.EntiNameEmb is None):
            traj_clsme_avg = traj_classeme.mean(dim=1) # shape == (n_enti,300)
        else:
            traj_clsme_avg = None
        
        pred_logits = self.prediction_head(pred_queries,att_matrx,proposal.cat_ids,traj_clsme_avg,enti2enco) # shpae == (n_querys, n_pred_cat)

        
        return pred_queries,pred_logits,att_matrx
    

    def prediction_head(self,pred_queries,att_matrx,cat_ids,enti_clsme,enti_feat):
        # pred_logits: shape == (n_querys, n_pred_cat) 
        # att_matrx:   shape == (2,n_querys,n_enti)
        # cat_ids: shape == (n_enti,)
        # enti_clsme.shape ==(n_enti,300)
        # self.bias_matrix shape == (n_enti_cat,n_enti_cat,n_pred_cat) # (81,81,51), including background

        pred_soid = torch.argmax(att_matrx,dim=-1)  # enti index,  shape == (2,n_querys)
        pred_socatid = cat_ids[pred_soid]  # enti categories, shape == (2,n_querys)
        pred_bias = self.bias_matrix[pred_socatid[0,:],pred_socatid[1,:],:] # shape == (n_querys,n_pred_cat)

        sub_feat = enti_feat[pred_soid[0,:],:]  # shape == (n_querys, dim_enti)
        obj_feat = enti_feat[pred_soid[1,:],:]  # shape == (n_querys, din_enti)

        if self.use_clsme:
            if self.EntiNameEmb is None:
                assert enti_clsme is not None
                sub_clsme = enti_clsme[pred_soid[0,:],:]  # shape == (n_querys, 300)
                obj_clsme = enti_clsme[pred_soid[1,:],:]  # shape == (n_querys, 300)
            else:
                assert enti_clsme is None
                sub_clsme = self.EntiNameEmb[pred_socatid[0,:],:]  # shape == (n_querys, 300)
                obj_clsme = self.EntiNameEmb[pred_socatid[1,:],:]  # shape == (n_querys, 300)
            pred_queries = torch.cat([pred_queries,sub_clsme,obj_clsme,sub_feat,obj_feat],dim=-1)  # shape == (n_querys,dim_pred+600+2*dim_enti)
        else:
            pred_queries = torch.cat([pred_queries,sub_feat,obj_feat],dim=-1)  # shape == (n_querys,dim_pred+2*dim_enti)
        pred_logits = self.fc_pred2logits(pred_queries)
        pred_logits = pred_logits + pred_bias
           
        return pred_logits
  
  
    def _forward_train(self,proposal_list,gt_graph_list):
        assert len(proposal_list) == len(gt_graph_list) 
        batch_size = len(proposal_list)
        mp_results = [self.encode2decode(proposal) for proposal in proposal_list]
        pred_queries,pred_logits,att_matrx = list(zip(*mp_results))
        # pred_queries.shape == (n_querys, dim_pred)
        # pred_logits: shape == (n_querys, n_pred_cat) 
        # att_matrx:   shape == (2,n_querys,max_enti)
        
        ## prepare gt
        gt_preds = [g.pred_cat_ids for g in gt_graph_list]
        # gt_entis = [g.traj_cat_ids for g in gt_graph_list]

        gt_adjs = [g.adj_matrix for g in gt_graph_list]
        # gt_adj_enti_align = [self.enti_viou_align(gt_adj,proposal,gt_graph) for gt_adj,proposal,gt_graph in zip(gt_adjs,proposal_list,gt_graph_list)]
        gt_adj_enti_align = []
        # viou_matrix = []
        gt_duras = []
        for gt_adj,proposal,gt_graph in zip(gt_adjs,proposal_list,gt_graph_list):
            gaea,viou_mat = self.enti_viou_align(gt_adj,proposal,gt_graph)
            gt_adj_enti_align.append(gaea)
            # viou_matrix.append(viou_mat)
        
            # gt_duras.append(
            #     gt_graph.pred_durations.float()       # shape == (n_gt_pred,2)
            # )
        
        

        ## bipartite_match & loss **without** temporal regression
        indices = []
        for pred_logit,gt_pred,att,gt_adj in zip(pred_logits,gt_preds,att_matrx,gt_adj_enti_align):
            index = self.bipartite_match(
                pred_logit,gt_pred,att,gt_adj
            )
            indices.append(index)


        loss_dict = self.loss(indices,
            pred_logits,att_matrx,
            gt_preds,gt_adj_enti_align,
        )

        total_loss = torch.stack(list(loss_dict.values())).sum()    # scalar tensor

        
        return total_loss, loss_dict

    
    def enti_viou_align(self,gt_adj,proposal,gt_graph):
        _,n_gt_pred,n_gt_enti = gt_adj.shape
        n_enti = proposal.num_proposals
        gt_adj_enti_align = torch.zeros(size=(2,n_gt_pred,n_enti),device=gt_adj.device)

        
        pr_trajbboxes,pr_trajduras = proposal.bboxes_list,proposal.traj_durations
        gt_trajbboxes,gt_trajduras = gt_graph.traj_bboxes,gt_graph.traj_durations      # gt 和 proposal 匹配的时候没有考虑 category
        gt_trajduras[:,1] -= 1 # 转为闭区间
        num_gt_enti = len(gt_trajbboxes)
        inter_dura,dura_mask = dura_intersection_ts(pr_trajduras,gt_trajduras)  # shape == (n_traj,n_gt_traj,2)
        
        inter_dura_p = inter_dura - pr_trajduras[:,0,None,None]  # convert to relative duration
        inter_dura_g = inter_dura - gt_trajduras[None,:,0,None]
        
        pids,gids = dura_mask.nonzero(as_tuple=True)  # row, col : pid,gid
        viou_matrix = torch.zeros_like(dura_mask,dtype=torch.float)
        for pid,gid in zip(pids.tolist(),gids.tolist()):
            dura_p = inter_dura_p[pid,gid,:]
            dura_g = inter_dura_g[pid,gid,:]
            bboxes_p = pr_trajbboxes[pid]
            bboxes_g = gt_trajbboxes[gid]
            viou_matrix[pid,gid] = vIoU_ts(bboxes_p,bboxes_g,dura_p,dura_g)
        
        viou_mask = viou_matrix > self.positive_vIoU_th
        # print(viou_mask)
        # if some gt_trajs has no proposal that > vIoU_th, we select the one with the max vIoU,
        #  ensure each gt_traj has at least one anchor (i.e., vIoUmask.sum() >= num_gt_enti)
        maxvIoU_prop_ind = torch.argmax(viou_matrix,dim=0)           # shape == (num_gt_enti,)
        sum0_mask = viou_mask.sum(dim=0) == 0                           # sahpe == (num_gt_enti,)
        maxvIoU_prop_ind = maxvIoU_prop_ind[sum0_mask]
        viou_mask[maxvIoU_prop_ind,sum0_mask] = 1                       # shape == (num_proposals,num_gt_enti)
        assert viou_mask.sum() >= num_gt_enti
        # print(viou_mask)
        # import time
        # time.sleep(1000)
        for pid, row in enumerate(viou_mask):
            if torch.sum(row) == 0:
                # gt_adj_enti_align[:,:,pid] = 0
                pass
            else:
                # gt_enti_ind = torch.argmax(row.float())
                gt_enti_ind = torch.argmax(viou_matrix[pid,:])
                gt_adj_enti_align[:,:,pid] = gt_adj[:,:,gt_enti_ind]
        
        return gt_adj_enti_align,viou_matrix

    @torch.no_grad()
    def bipartite_match(self,
            pred_logit,gt_pred,att_matrx,gt_adj_enti_align,
        ):
        # pred_logit.shape == (n_querys,n_pred_cats)
        # gt_pred.shape == (n_gt_pred,)
        # pred_reg.shape == (n_querys,2)
        # reg_target.shape == (n_querys,n_gt_pred,2)
        _,n_gt_pred,n_enti = gt_adj_enti_align.shape


        pred_logit = pred_logit[:,:,None].repeat(1,1,n_gt_pred)  # shape == (n_querys,n_pred_cats,n_gt_preds)
        gt_pred = gt_pred[None,:].repeat(self.num_querys,1)     # shape == (n_querys,n_gt_preds)

        cost_cls = F.cross_entropy(pred_logit,gt_pred,reduction='none')  # shape == (n_querys,n_gt_preds)
        

        att_matrx = att_matrx[:,:,None,:].repeat(1,1,n_gt_pred,1)
        gt_adj_enti_align = gt_adj_enti_align[:,None,:,:].repeat(1,self.num_querys,1,1)
        cost_adj = F.binary_cross_entropy(att_matrx,gt_adj_enti_align,reduction='none')  # shape == (2,n_querys,n_gt_pred,n_enti)
        cost_adj = cost_adj.mean(dim=[0,-1])    # shape == (n_querys, n_gt_pred)
        
        
        cost_cls *= self.cost_factor["classification"]
        cost_adj *= self.cost_factor["adj_matrix"]
        # print(cost_cls,"cls")
        # print(cost_adj,"adj")
        # import time
        # time.sleep(1000)

        cost_all = cost_cls + cost_adj
        index = linear_sum_assignment(cost_all.cpu())

        return index


    def loss(self,indices,
            pred_logits,att_matrx,gt_preds,gt_adj
        ):
        # indices == [(idx_anchor1,idx_gt1),...,(idx_anchorN,idx_gtN)]
        batch_size = len(indices)
        
        ## 1.classification loss
        gt_targets = []
        for i,(idx_anchor,idx_gt) in enumerate(indices):
            gt_preds_align = torch.zeros(size=(self.num_querys,),dtype=torch.long,device=gt_preds[0].device)
            gt_preds_align[idx_anchor] = gt_preds[i][idx_gt]
            gt_targets.append(gt_preds_align)
        gt_targets = torch.cat(gt_targets)
        pos_mask = gt_targets != 0 
        neg_mask = torch.logical_not(pos_mask)  # shape == (N*n_querys,)
        pred_logits_ = torch.cat(pred_logits,dim=0)
        cls_loss = F.cross_entropy(pred_logits_,gt_targets,reduction='none') # shape == (N*anchors,)
        cls_pos = cls_loss[pos_mask].mean()
        if neg_mask.sum() > 0:
            cls_neg = cls_loss[neg_mask].mean()
        else:
            cls_neg = torch.zeros_like(cls_pos)
        
        # gt_targets_binary = torch.zeros_like(pred_logits)
        # gt_targets_binary[range(batch_size*self.num_querys),gt_targets] = 1
        # focal_loss = sigmoid_focal_loss_jit(
        #     pred_logits,
        #     gt_targets_binary,
        #     alpha=self.alpha,
        #     gamma=self.gamma,
        #     reduction='none'
        # )
        # cls_pos = focal_loss[pos_mask].sum(dim=-1).mean()  # focal_loss[pos_mask] == focal_loss[pos_mask[:,None].repeat(1,n_cats)]  # shape == (n_pos,n_cats)
        # if neg_mask.sum() > 0:
        #     cls_neg = focal_loss[neg_mask].sum(dim=-1).mean()
        # else:
        #     cls_neg = torch.zeros_like(cls_pos)
        
        # diverse
        ## 3. adj_matrix loss
        # att_matrx.shape == (2,n_querys,n_enti)
        # gt_adj.shape == (2,n_gt_pred,n_enti)
        att_matrx_ = [att_matrx[i][:,idx_anchor,:].reshape(2,-1) for i,(idx_anchor,_) in enumerate(indices)]  # shape == (2,n_positive,n_enti) --> (2,n_positive*n_enti)
        gt_adj = [gt_adj[i][:,idx_gt,:].reshape(2,-1) for i,(_,idx_gt) in enumerate(indices)]  # shape == (2,n_positive,n_enti)
        att_matrx_ = torch.cat(att_matrx_,dim=-1)
        gt_adj = torch.cat(gt_adj,dim=-1)
        bce_weight = torch.ones_like(gt_adj)
        gt_adj_mask = gt_adj.type(torch.bool)
        bce_weight[~gt_adj_mask] *= self.neg_weight
        adj_loss = F.binary_cross_entropy(att_matrx_,gt_adj,weight=bce_weight,reduction='none').mean()
        



        cls_pos *= self.loss_factor["classification"]
        cls_neg *= self.loss_factor["classification"]
        adj_loss *= self.loss_factor["adj_matrix"]
        loss_dict = {
            "cls_pos":cls_pos,
            "cls_neg":cls_neg,
            "adj":adj_loss
        }
        return loss_dict


    def construct_triplet(self,proposal,pred_logits,att_matrx):
        # pred_cs.shape == (n_pred,2) center span
        

        pred_probs = torch.softmax(pred_logits,dim=-1)
        pred_scores,pred_catids = torch.topk(pred_probs,self.topk,dim=-1)  # shape == (n_anchors,k)
        pred_scores = pred_scores.reshape(-1) # shape == (n_ac*k,) # flatten as concatenate each row
        pred_catids = pred_catids.reshape(-1) # shape == (n_ac*k,)
        predquery_ids = torch.tensor(list(range(self.num_anchors)),device=pred_catids.device)  # shape == (n_ac,)
        predquery_ids = torch.repeat_interleave(predquery_ids,self.topk) # shape == (n_ac*k,)
        
        traj_duras = proposal.traj_durations.clone() # shape == (n_enti,2)
        n_traj = traj_duras.shape[0]

        enti_scores = proposal.scores            # shape == (n_enti,)
        enti_catids = proposal.cat_ids           # shape == (n_enti,)
        
        pred2so_ids = torch.argmax(att_matrx,dim=-1).t()  # enti index,  shape == (n_ac,2)
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
        predquery_ids=predquery_ids[pos_pred_index]  # shape == (n_pos,)
        
        
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
        uniq_query_ids = predquery_ids[uniq_triplet_ids]    # shape == (n_unique,)
                               
        uniq_dura_inters = dura_inters[uniq_quintuples[:,3],uniq_quintuples[:,4],:] # shape == (n_unique,2)
        #TODO sort by socre and select top100?


        # filter out triplets whose pred_cat is __background__
        mask = uniq_quintuples[:,0] != 0
        uniq_quintuples = uniq_quintuples[mask,:]
        uniq_scores = uniq_scores[mask,:]
        uniq_query_ids = uniq_query_ids[mask]
        uniq_dura_inters = uniq_dura_inters[mask,:]


        ret = (
            uniq_quintuples,    # shape == (n_unique,5)
            uniq_scores,        # shape == (n_unique,3)
            uniq_dura_inters,   # shape == (n_unique,2) 
            uniq_query_ids,     # shape == (n_unique,)
        )

        return ret


 