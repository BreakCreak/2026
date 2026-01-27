import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math
import numpy as np
torch.set_printoptions(profile="full")

class MixedExpert(nn.Module):
    def __init__(self, c_in, c_out=512):  # 输出维度改为512
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv1d(2*c_in, c_out, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
        )
        
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化 [B, C, 1]
            nn.Conv1d(c_out, c_out // 16, 1, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out // 16, c_out, 1, bias=False),  # 升维回原维度
            nn.Sigmoid()  # 使用Sigmoid激活得到注意力权重
        )

    def forward(self, rgb, flow):
        x = torch.cat([rgb, flow], dim=1)  # [B, 2*C_in, T] -> [B, 2*c_in, T]
        x = self.fusion(x)  # [B, c_out, T]
        
        # 应用通道注意力
        att_weights = self.channel_att(x)  # [B, c_out, 1]
        x = x * att_weights  # [B, c_out, T] * [B, c_out, 1] -> [B, c_out, T]
        
        return x

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

        # 特征对齐模块
        self.align_flow = nn.Conv1d(512, 512, 1)  # Flow特征对齐到RGB特征空间

        # 混合专家模块 - 输出维度为512
        self.mixed_expert = MixedExpert(512, 512)  # 输入512,512，输出512
        self.cls_mixed = nn.Conv1d(512, 1, 1)  # 混合专家分支的分类头，输入512
        self.cls_branch1 = nn.Conv1d(512, 1, 1)  # Branch1的分类头，输入512

        # 门控模块 - 用于选择RGB+Flow分支还是Mixed分支 (二值选择)
        self.gate_logits = nn.Sequential(
            nn.Conv1d(512 + 512, 512, 3, padding=1),  # RGB+Flow (512) + Mixed (512) = 1024维
            nn.ReLU(),
            nn.Conv1d(512, 2, 1),  # 输出2个门控logits
        )
        
        # Gumbel-Softmax的温度参数
        self.tau = 1.0  # 温度参数，控制输出的平滑程度

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        emb_flow = self.action_module_flow(input[:, 1024:, :])
        emb_rgb = self.action_module_rgb(input[:, :1024, :])

        # RGB + Flow 相加 (维度为512)，对Flow特征进行对齐
        emb_branch1 = emb_rgb + self.align_flow(emb_flow)  # [B, 512, T]

        # 混合专家分支 - 输出维度为512
        emb_mixed = self.mixed_expert(
            emb_rgb,  # RGB branch (已处理的特征)
            emb_flow   # Flow branch (已处理的特征)
        )  # [B, 512, T]

        # 将Branch1和Mixed分支的特征拼接用于门控
        combined_for_gating = torch.cat([emb_branch1, emb_mixed], dim=1)  # [B, 512+512=1024, T]

        # 门控 - 选择RGB+Flow分支还是Mixed分支 (二值选择)
        gate_logits = self.gate_logits(combined_for_gating)  # [B, 2, T]
        
        # 使用Gumbel-Softmax进行门控选择，支持梯度传递
        gate_weights = F.gumbel_softmax(
            gate_logits,
            tau=self.tau,
            hard=not self.training,  # 训练时使用软选择，推理时使用硬选择
            dim=1
        )  # [B, 2, T]
        
        # 在推理时，我们仍然可以通过argmax获得硬选择
        if not self.training:
            # 推理时使用硬选择，但训练时保持可微分
            gate_weights_hard = torch.zeros_like(gate_logits)
            max_indices = torch.argmax(gate_logits, dim=1, keepdim=True)  # [B, 1, T]
            gate_weights_hard.scatter_(1, max_indices, 1.0)  # 设置最大值位置为1
            branch1_w, branch2_w = gate_weights_hard[:, 0:1, :], gate_weights_hard[:, 1:2, :]
        else:
            # 训练时使用Gumbel-Softmax的输出
            branch1_w, branch2_w = gate_weights[:, 0:1, :], gate_weights[:, 1:2, :]

        # 分支1: RGB + Flow 相加（分别分类再加和）
        action_branch1 = torch.sigmoid(self.cls_rgb(emb_rgb)).squeeze(1) + torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)  # [B, T]，使用原始的分类头

        # 分支2: Mixed 分支
        action_branch2 = torch.sigmoid(self.cls_mixed(emb_mixed)).squeeze(1)  # [B, T]

        # 使用门控权重选择两个分支
        actionness_selected = branch1_w.squeeze(1) * action_branch1 + branch2_w.squeeze(1) * action_branch2

        # 保存用于对比学习的嵌入
        embedding_flow = emb_flow.permute(0, 2, 1)
        embedding_rgb = emb_rgb.permute(0, 2, 1)
        embedding_mixed = emb_mixed.permute(0, 2, 1)
        embedding_branch1 = emb_branch1.permute(0, 2, 1)  # RGB+Flow相加后的嵌入

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
            embedding_branch1,  # 新增：RGB+Flow相加后的嵌入
            gate_weights,
            action_branch1,  # 新增：分支1的actionness
            action_branch2   # 新增：分支2的actionness
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

        cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb, embedding_mixed, embedding_branch1, gate_weights, action_branch1, action_branch2 = self.actionness_module(x)

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_median1 = np.median(aness_np1, 1, keepdims=True)
        aness_bin1 = np.where(aness_np1 > aness_median1, 1.0, 0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_median2 = np.median(aness_np2, 1, keepdims=True)
        aness_bin2 = np.where(aness_np2 > aness_median2, 1.0, 0.0)

        # 为分支1和分支2也计算二值化掩码，用于各自的对比学习
        action_branch1_np = action_branch1.cpu().detach().numpy()
        action_branch1_bin = np.where(action_branch1_np > np.median(action_branch1_np, 1, keepdims=True), 1.0, 0.0)
        
        action_branch2_np = action_branch2.cpu().detach().numpy()
        action_branch2_bin = np.where(action_branch2_np > np.median(action_branch2_np, 1, keepdims=True), 1.0, 0.0)

        # actionness = actionness1 + actionness2

        CA, CB = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        CAr, CBr = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        CAf, CBf = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        # Mixed branch contrastive learning (混合专家分支)
        CAm, CBm = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_mixed, k_C
        )
        IAm, IBm = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_mixed, k_I
        )

        # Branch1 contrastive learning (RGB+Flow相加分支)
        CAb1, CBb1 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_branch1, k_C
        )
        IBb1, IBb1_ = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness1, embedding_branch1, k_I
        )

        # 为混合专家分支添加额外的对比学习 - 使用门控权重来指导学习
        CAm2, CBm2 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_mixed, k_C
        )
        IAm2, IBm2 = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_mixed, k_I
        )

        # 为RGB+Flow相加分支添加额外的对比学习
        CAb1_2, CBb1_2 = self.consistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_branch1, k_C
        )
        IBb1_2, IBb1_2_ = self.Inconsistency_snippets_mining1(
            aness_bin1, aness_bin2, actionness2, embedding_branch1, k_I
        )

        # 为每个分支单独计算的对比对（用于门控反馈）
        # Branch1单独的对比对
        CAb1_ind, CBb1_ind = self.consistency_snippets_mining1(
            action_branch1_bin, action_branch1_bin, action_branch1, embedding_branch1, k_C
        )
        IBb1_ind, IBb1_ind_ = self.Inconsistency_snippets_mining1(
            action_branch1_bin, action_branch1_bin, action_branch1, embedding_branch1, k_I
        )
        
        # Branch2单独的对比对
        CAm_ind, CBm_ind = self.consistency_snippets_mining1(
            action_branch2_bin, action_branch2_bin, action_branch2, embedding_mixed, k_C
        )
        Im_ind, Im_ind_ = self.Inconsistency_snippets_mining1(
            action_branch2_bin, action_branch2_bin, action_branch2, embedding_mixed, k_I
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

        # 返回所有需要的嵌入和门控权重
        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2