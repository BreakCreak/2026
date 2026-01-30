import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math
import numpy as np
torch.set_printoptions(profile="full")

class BaseModel_Flow_Only(nn.Module):
    def __init__(self, len_feature, num_classes, config=None):
        super(BaseModel_Flow_Only, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        # 移除 RGB 分支
        # self.action_module_rgb = ...
        # self.cls_rgb = ...

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls_flow = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)

        # 移除混合专家模块
        # self.mixed_expert = ...
        # self.cls_mixed = ...
        # self.gate_logits = ...
        # self.tau = ...

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        # 仅处理Flow特征
        emb_flow = self.action_module_flow(input[:, 1024:, :])
        
        # 移除RGB处理
        # emb_rgb = self.action_module_rgb(input[:, :1024, :])
        emb_rgb = emb_flow  # 使用Flow作为占位符

        # 仅使用Flow分支
        emb_branch1 = emb_flow  # [B, 512, T]

        # 移除混合专家分支
        emb_mixed = emb_flow  # 使用Flow作为替代

        # 移除门控机制 - 直接使用Flow分支
        gate_weights = torch.zeros(input.size(0), 2, input.size(2), device=input.device)
        gate_weights[:, 1, :] = 1.0  # 全部选择Flow分支

        # 仅Flow分支的动作性预测
        action_branch1 = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)  # [B, T]
        action_branch2 = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)  # [B, T] (复制作为占位符)

        # 仅使用Flow分支
        actionness_selected = action_branch1

        # 保存用于对比学习的嵌入
        embedding_flow = emb_flow.permute(0, 2, 1)
        embedding_rgb = torch.zeros_like(emb_flow.permute(0, 2, 1))  # 占位符
        embedding_mixed = emb_flow.permute(0, 2, 1)  # 占位符
        embedding_branch1 = emb_branch1.permute(0, 2, 1)

        emb = self.base_module(input)
        embedding = emb.permute(0, 2, 1)
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = cas.sum(dim=2)
        actionness1 = torch.sigmoid(actionness1)

        action_flow = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)
        action_rgb = torch.zeros_like(action_flow)  # 占位符

        # 最终的actionness2是Flow分支的结果
        actionness2 = actionness_selected

        return (
            cas,
            action_flow,
            action_rgb,
            actionness1,
            actionness2,
            embedding,
            embedding_flow,
            embedding_rgb,
            embedding_mixed,
            embedding_branch1,
            gate_weights,
            action_branch1,
            action_branch2
        )


class E1_Flow_Only(nn.Module):
    def __init__(self, cfg):
        super(E1_Flow_Only, self).__init__()
        self.len_feature = 2048
        self.num_classes = 20

        self.actionness_module = BaseModel_Flow_Only(self.len_feature, self.num_classes, cfg)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_C = 20
        self.r_I = 20

        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def consistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness1, embeddings, k_easy):
        x = aness_bin1 + aness_bin2
        select_idx_act = actionness1.new_tensor(np.where(x == 2, 1, 0))

        actionness_act = actionness1 * select_idx_act

        select_idx_bg = actionness1.new_tensor(np.where(x == 0, 1, 0))

        actionness_rev = torch.max(actionness1, dim=1, keepdim=True)[0] - actionness1
        actionness_bg = actionness_rev * select_idx_bg

        easy_act = self.select_topk_embeddings(actionness_act, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_bg, embeddings, k_easy)

        return easy_act, easy_bkg

    def Inconsistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness1, embeddings, k_hard):
        x = aness_bin1 + aness_bin2
        idx_region_inner = actionness1.new_tensor(np.where(x == 1, 1, 0))
        aness_region_inner = actionness1 * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        actionness_rev = torch.max(actionness1, dim=1, keepdim=True)[0] - actionness1
        aness_region_outer = actionness_rev * idx_region_inner
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def forward(self, x):
        num_segments = x.shape[1]
        k_C = num_segments // self.r_C
        k_I = num_segments // self.r_I

        cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb, embedding_mixed, embedding_branch1, gate_weights, action_branch1, action_branch2 = self.actionness_module(x)

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_median1 = np.median(aness_np1, 1, keepdims=True)
        aness_bin1 = np.where(aness_np1 > aness_median1, 1.0, 0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_median2 = np.median(aness_np2, 1, keepdims=True)
        aness_bin2 = np.where(aness_np2 > aness_median2, 1.0, 0.0)

        action_branch1_np = action_branch1.cpu().detach().numpy()
        action_branch1_bin = np.where(action_branch1_np > np.median(action_branch1_np, 1, keepdims=True), 1.0, 0.0)
        
        action_branch2_np = action_branch2.cpu().detach().numpy()
        action_branch2_bin = np.where(action_branch2_np > np.median(action_branch2_np, 1, keepdims=True), 1.0, 0.0)

        CA, CB = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        # RGB分支对比对使用占位符
        CAr = torch.zeros_like(CA)
        CBr = torch.zeros_like(CB)
        IAr = torch.zeros_like(IA)
        IBr = torch.zeros_like(IB)

        CAf, CBf = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        # Mixed branch contrastive learning (使用Flow分支作为占位符)
        CAm, CBm = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_flow, k_C
        )
        IAm, IBm = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_flow, k_I
        )

        # Branch1 contrastive learning (Flow分支)
        CAb1, CBb1 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_branch1, k_C
        )
        IBb1, IBb1_ = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_branch1, k_I
        )

        # 为混合专家分支添加额外的对比学习 - 使用Flow作为占位符
        CAm2, CBm2 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_flow, k_C
        )
        IAm2, IBm2 = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_flow, k_I
        )

        # 为RGB+Flow相加分支添加额外的对比学习
        CAb1_2, CBb1_2 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_branch1, k_C
        )
        IBb1_2, IBb1_2_ = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_branch1, k_I
        )

        # 为每个分支单独计算的对比对（用于门控反馈）
        CAb1_ind, CBb1_ind = self.consistency_snippets_mining1(
            action_branch1_bin, action_branch1_bin, action_branch1, embedding_branch1, k_C
        )
        IBb1_ind, IBb1_ind_ = self.Inconsistency_snippets_mining1(
            action_branch1_bin, action_branch1_bin, action_branch1, embedding_branch1, k_I
        )
        
        # Branch2单独的对比对 (使用Flow作为占位符)
        CAm_ind, CBm_ind = self.consistency_snippets_mining1(
            action_branch2_bin, action_branch2_bin, action_branch2, embedding_flow, k_C
        )
        Im_ind, Im_ind_ = self.Inconsistency_snippets_mining1(
            action_branch2_bin, action_branch2_bin, action_branch2, embedding_flow, k_I
        )

        contrast_pairs = {
            'CA': CA,
            'CB': CB,
            'IA': IA,
            'IB': IB
        }

        contrast_pairs_r = {
            'CA': CAr,
            'CB': CBr,
            'IA': IAr,
            'IB': IBr
        }

        contrast_pairs_f = {
            'CA': CAf,
            'CB': CBf,
            'IA': IAf,
            'IB': IBf
        }

        contrast_pairs_m = {'CA': CAm, 'CB': CBm, 'IA': IAm, 'IB': IBm}
        contrast_pairs_m2 = {'CA': CAm2, 'CB': CBm2, 'IA': IAm2, 'IB': IBm2}
        contrast_pairs_b1 = {'CA': CAb1, 'CB': CBb1, 'IA': IBb1, 'IB': IBb1_}
        contrast_pairs_b1_2 = {'CA': CAb1_2, 'CB': CBb1_2, 'IA': IBb1_2, 'IB': IBb1_2_}
        contrast_pairs_b1_ind = {'CA': CAb1_ind, 'CB': CBb1_ind, 'IA': IBb1_ind, 'IB': IBb1_ind_}
        contrast_pairs_m_ind = {'CA': CAm_ind, 'CB': CBm_ind, 'IA': Im_ind, 'IB': Im_ind_}

        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2