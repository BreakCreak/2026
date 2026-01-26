import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math
import numpy as np
torch.set_printoptions(profile="full")

class MixedExpert(nn.Module):
    def __init__(self, c_in, c_out=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv1d(2*c_in, c_out, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
        )

    def forward(self, rgb, flow):
        x = torch.cat([rgb, flow], dim=1)  # [B, 2*C_in, T]
        return self.fusion(x)

class BaseModel(nn.Module):
    def __init__(self, len_feature, num_classes, config=None):
        super(BaseModel, self).__init__()
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

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_rgb = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_flow = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)  # 0.5

        # 混合专家模块 - 单一混合专家
        self.mixed_expert = MixedExpert(1024, 512)
        self.cls_mixed = nn.Conv1d(512, 1, 1)

        # 门控模块 - 用于选择RGB+Flow分支还是Mixed分支
        self.gate_module = nn.Sequential(
            nn.Conv1d(512 * 3, 512, 3, padding=1),  # RGB + Flow + Mixed = 3*512维特征拼接
            nn.ReLU(),
            nn.Conv1d(512, 2, 1),  # 输出2个门控权重
            nn.Softmax(dim=1)
        )

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        emb_flow = self.action_module_flow(input[:, 1024:, :])
        emb_rgb = self.action_module_rgb(input[:, :1024, :])

        # 混合专家分支 - 单一混合专家
        emb_mixed = self.mixed_expert(
            input[:, :1024, :],  # RGB branch
            input[:, 1024:, :]   # Flow branch
        )

        # 将RGB、Flow和Mixed分支的特征拼接用于门控
        combined_for_gating = torch.cat([emb_rgb, emb_flow, emb_mixed], dim=1)

        # 门控 - 选择RGB+Flow分支还是Mixed分支
        gate_weights = self.gate_module(combined_for_gating)
        if inference:
            # 软化门控
            gate_weights = gate_weights * 0.7 + 0.3 / 2

        # 分解门控权重
        branch1_w, branch2_w = gate_weights[:, 0:1, :], gate_weights[:, 1:2, :]

        # Branch 1: RGB + Flow 相加
        emb_branch1 = emb_rgb + emb_flow  # 直接相加
        action_branch1 = torch.sigmoid(self.cls_mixed(emb_branch1)).squeeze(1)

        # Branch 2: Mixed 分支
        emb_branch2 = emb_mixed
        action_branch2 = torch.sigmoid(self.cls_mixed(emb_branch2)).squeeze(1)

        # 使用门控权重选择两个分支
        actionness_selected = branch1_w.squeeze(1) * action_branch1 + branch2_w.squeeze(1) * action_branch2

        embedding_flow = emb_flow.permute(0, 2, 1)
        embedding_rgb = emb_rgb.permute(0, 2, 1)
        embedding_mixed = emb_mixed.permute(0, 2, 1)

        # action_flow = torch.sigmoid(self.cls_flow(emb_flow))
        # action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb))

        emb = self.base_module(input)
        embedding = emb.permute(0, 2, 1)
        # emb = self.dropout(emb)
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = cas.sum(dim=2)
        actionness1 = torch.sigmoid(actionness1)

        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb)).squeeze(1)
        action_flow = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)

        # 最终的actionness2是门控选择的结果
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
            gate_weights
        )


class AICL(nn.Module):
    def __init__(self, cfg):
        super(AICL, self).__init__()
        self.len_feature = 2048
        self.num_classes = 20

        self.actionness_module = BaseModel(self.len_feature, self.num_classes, cfg)

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
        # print(torch.min(torch.sum(select_idx_act, dim=-1)))

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

        cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb, embedding_mixed, gate_weights = self.actionness_module(x)

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_median1 = np.median(aness_np1, 1, keepdims=True)
        aness_bin1 = np.where(aness_np1 > aness_median1, 1.0, 0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_median2 = np.median(aness_np2, 1, keepdims=True)
        aness_bin2 = np.where(aness_np2 > aness_median2, 1.0, 0.0)

        # actionness = actionness1 + actionness2

        CA, CB = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        CAr, CBr = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        CAf, CBf = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        # Mixed branch contrastive learning
        CAm, CBm = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_mixed, k_C
        )
        IAm, IBm = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_mixed, k_I
        )

        # 为混合专家分支添加额外的对比学习 - 使用门控权重来指导学习
        # 当branch1_w较大（直接相加效果好）时，可能意味着混合专家需要改进
        # 当branch2_w较大（混合专家效果好）时，说明混合专家学习得好
        CAm2, CBm2 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_mixed, k_C
        )
        IAm2, IBm2 = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_mixed, k_I
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

        # 返回 embedding_mixed 和 gate_weights 以供训练时使用
        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed