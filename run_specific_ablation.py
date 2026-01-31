"""
单独运行特定的消融实验
使用方法: python run_specific_ablation.py --experiment_type E1_RGB
"""

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from collections import OrderedDict
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_thumos import Config, parse_args, class_dict

# 导入各种消融实验模型
from models.model import AICL  # E5: 完整模型
from models.e1_rgb_only_model import E1_RGB_Only
from models.e1_flow_only_model import E1_Flow_Only
from models.e1_joint_only_model import E1_Joint_Only
from models.e2_no_mixed_expert_model import E2_No_MixedExpert
from models.e3_mixed_expert_no_gate_model import E3_MixedExpert_NoGate
from models.e4_gate_off_model import E4_GateOff
from models.e6_no_gate_regularization_model import E6_NoGateRegularization

def set_seed(seed):
    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss


def gate_entropy_loss(gate_weights):
    p = gate_weights + 1e-6
    entropy = -torch.sum(p * torch.log(p), dim=1)
    return entropy.mean()


def branch_balance_loss(gate_weights):
    avg_activation = torch.mean(gate_weights, dim=2)
    batch_avg = torch.mean(avg_activation, dim=0)
    ideal_ratio = 0.5
    balance_loss = torch.mean((batch_avg - ideal_ratio) ** 2)
    return balance_loss


class SpecificAblationTrainer():
    def __init__(self, config, experiment_type="E5"):
        # config
        self.config = config
        self.experiment_type = experiment_type

        # 根据实验类型选择模型
        if experiment_type == "E1_RGB":
            self.net = E1_RGB_Only(config)
        elif experiment_type == "E1_Flow":
            self.net = E1_Flow_Only(config)
        elif experiment_type == "E1_Joint":
            self.net = E1_Joint_Only(config)
        elif experiment_type == "E2_NoME":
            self.net = E2_No_MixedExpert(config)
        elif experiment_type == "E3_ME_NoGate":
            self.net = E3_MixedExpert_NoGate(config)
        elif experiment_type == "E4_GateOff":
            self.net = E4_GateOff(config)
        elif experiment_type == "E5_Full":
            self.net = AICL(config)  # 完整模型
        elif experiment_type == "E6_NoReg":
            self.net = E6_NoGateRegularization(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
            
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # parameters
        self.best_mAP = -1
        self.step = 0
        self.total_loss_per_epoch = 0

        # 为不同的实验类型设置不同的损失权重
        if experiment_type in ["E1_RGB", "E1_Flow", "E1_Joint"]:
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.0
            self.lambda_gate_balance = 0.0
            self.lambda_gate_feedback = 0.0
        elif experiment_type == "E4_GateOff":
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.01
            self.lambda_gate_balance = 0.02
            self.lambda_gate_feedback = 0.0
        elif experiment_type in ["E5_Full", "E6_NoReg"]:
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.01 if experiment_type == "E5_Full" else 0.0
            self.lambda_gate_balance = 0.02 if experiment_type == "E5_Full" else 0.0
            self.lambda_gate_feedback = 0.05 if experiment_type == "E5_Full" else 0.0
        else:
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.01
            self.lambda_gate_balance = 0.02
            self.lambda_gate_feedback = 0.05

    def calculate_gate_feedback_loss(self, gate_weights, action_branch1, action_branch2, topk_indices):
        batch_size = gate_weights.size(0)
        
        if len(topk_indices.shape) != 2 or topk_indices.shape[0] != action_branch1.shape[0]:
            _, temp_topk_indices = torch.topk(action_branch1, min(10, action_branch1.size(1)), dim=1)
            topk_indices = temp_topk_indices
        
        action_branch1_topk = torch.gather(action_branch1, 1, topk_indices)
        action_branch2_topk = torch.gather(action_branch2, 1, topk_indices)
        
        branch1_quality = action_branch1_topk.mean(dim=1, keepdim=True)
        branch2_quality = action_branch2_topk.mean(dim=1, keepdim=True)
        
        mask_branch1_better = branch1_quality > branch2_quality
        
        relative_advantage = torch.zeros((batch_size, 2, 1), device=gate_weights.device)
        relative_advantage[:, 0, :] = mask_branch1_better.float()
        relative_advantage[:, 1, :] = (~mask_branch1_better).float()
        
        expanded_indices = topk_indices.unsqueeze(1).expand(-1, 2, -1)
        gate_weights_topk = torch.gather(gate_weights, 2, expanded_indices)
        avg_gate_weights = gate_weights_topk.mean(dim=2, keepdim=True)
        
        gate_feedback_loss = F.mse_loss(avg_gate_weights, relative_advantage)
        return gate_feedback_loss

    def calculate_all_losses(self, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, cas_top, label, topk_indices, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2):
        self.contrastive_criterion = ContrastiveLoss()
        
        L_c = self.contrastive_criterion(contrast_pairs)
        L_r = self.contrastive_criterion(contrast_pairs_r)
        L_f = self.contrastive_criterion(contrast_pairs_f)
        L_m = self.contrastive_criterion(contrast_pairs_m)
        L_m2 = self.contrastive_criterion(contrast_pairs_m2)
        L_b1 = self.contrastive_criterion(contrast_pairs_b1)
        L_b1_2 = self.contrastive_criterion(contrast_pairs_b1_2)
        L_b1_ind = self.contrastive_criterion(contrast_pairs_b1_ind)
        L_m_ind = self.contrastive_criterion(contrast_pairs_m_ind)
        
        loss_contrastive = L_c + L_r + L_f + 0.5 * L_m + 0.3 * L_m2 + 0.5 * L_b1 + 0.3 * L_b1_2 + 0.4 * L_b1_ind + 0.4 * L_m_ind

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = F.mse_loss(action_flow, action_rgb)  # 合并为一次MSE
        action_consistent_loss = F.mse_loss(actionness1, actionness2)
    
        gate_ent_loss = gate_entropy_loss(gate_weights) if self.lambda_gate_entropy > 0 else torch.tensor(0.0, device=gate_weights.device)
        gate_balance_loss = branch_balance_loss(gate_weights) if self.lambda_gate_balance > 0 else torch.tensor(0.0, device=gate_weights.device)
        gate_feedback_loss = self.calculate_gate_feedback_loss(gate_weights, action_branch1, action_branch2, topk_indices) if self.lambda_gate_feedback > 0 else torch.tensor(0.0, device=gate_weights.device)

        # 调整后的对比损失权重，移除重复项，降低独立对比损失权重
        adjusted_contrastive_loss = L_c + L_r + L_f + 0.5 * L_m  # 移除了 L_m2 和 L_b1_2
        
        cost = (base_loss + 
                class_agnostic_loss + 
                5*modality_consistent_loss + 
                self.lambda_contrastive*adjusted_contrastive_loss + 
                0.1*action_consistent_loss + 
                self.lambda_gate_entropy * gate_ent_loss + 
                self.lambda_gate_balance * gate_balance_loss + 
                self.lambda_gate_feedback * gate_feedback_loss)

        return cost

    def forward_pass(self, _data):
        cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2 = self.net(_data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.unsqueeze(2).detach(),
                                                              action_rgb.unsqueeze(2))

        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2

    def train(self):
        set_seed(self.config.seed)

        for epoch in range(self.config.num_epochs):
            for i, (_data, _label, temp_anno, _, _) in enumerate(self.train_loader):
                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2 = self.forward_pass(_data)

                # calculate pseudo target
                cls_agnostic_gt = torch.zeros((batch_size, 1, self.config.num_segments)).cuda()
                
                for b in range(batch_size):
                    label_indices_b = torch.nonzero(_label[b, :])[:,0]
                    topk_indices_b = topk_indices[b, :, label_indices_b] if len(label_indices_b) > 0 else topk_indices[b, :, :1]
                    for gt_i in range(min(len(label_indices_b), topk_indices_b.shape[0])):
                        cls_agnostic_gt[b, 0, topk_indices_b[gt_i]] = 1

                # losses
                cost = self.calculate_all_losses(contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, cas_top, _label, topk_indices, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2)

                cost.backward()
                self.optimizer.step()

                if i % 10 == 0:  # 每10个batch打印一次
                    print(f"[{self.experiment_type}] Epoch {epoch}, Batch {i}, Loss: {cost.item():.4f}")

                if i % 100 == 0 and i > 0:  # 每100个batch进行一次评估
                    with torch.no_grad():
                        self.net.eval()
                        mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                        self.net.train()

                        if mean_ap > self.best_mAP:
                            self.best_mAP = mean_ap
                            model_filename = f"{self.experiment_type}_CAS_Only.pkl"
                            torch.save(self.net.state_dict(), os.path.join(self.config.model_path, model_filename))

                        print(f"[{self.experiment_type}] Epoch {epoch}, Batch {i}, mAP: {mean_ap*100:.2f}, Acc: {test_acc*100:.2f}, Best mAP: {self.best_mAP*100:.2f}")


def main():
    args = parse_args()
    config = Config(args)
    
    # 确保模型路径存在
    os.makedirs(config.model_path, exist_ok=True)
    
    # 从命令行参数获取实验类型
    experiment_type = getattr(args, 'experiment_type', 'E5_Full')  # 默认为完整模型
    
    print(f"开始运行实验: {experiment_type}")
    trainer = SpecificAblationTrainer(config, experiment_type)
    trainer.train()


if __name__ == '__main__':
    main()