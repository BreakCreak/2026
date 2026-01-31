"""
综合损失函数消融实验
专注于混合专家和门控机制的损失函数参数消融
包含不同权重组合的详细测试
"""

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_thumos import Config, parse_args, class_dict
from models.model import AICL

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

class ComprehensiveLossAblationTrainer():
    def __init__(self, config, experiment_name, lambda_contrastive=0.1, lambda_gate_entropy=0.01, lambda_gate_balance=0.02, lambda_gate_feedback=0.2):
        self.config = config
        self.experiment_name = experiment_name

        self.net = AICL(config)
        self.net = self.net.cuda()

        self.train_loader, self.test_loader = get_dataloaders(self.config)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        self.best_mAP = -1
        self.step = 0
        
        # 损失函数权重参数
        self.lambda_contrastive = lambda_contrastive
        self.lambda_gate_entropy = lambda_gate_entropy
        self.lambda_gate_balance = lambda_gate_balance
        self.lambda_gate_feedback = lambda_gate_feedback

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
        
        # 调整后的对比损失权重，移除重复项
        adjusted_contrastive_loss = L_c + L_r + L_f + 0.5 * L_m  # 移除了 L_m2 和 L_b1_2

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = F.mse_loss(action_flow, action_rgb)  # 合并为一次MSE
        action_consistent_loss = F.mse_loss(actionness1, actionness2)

        gate_ent_loss = gate_entropy_loss(gate_weights)
        gate_balance_loss = branch_balance_loss(gate_weights)
        gate_feedback_loss = self.calculate_gate_feedback_loss(gate_weights, action_branch1, action_branch2, topk_indices)

        # 使用传入的权重参数
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
                    print(f"[{self.experiment_name}] Epoch {epoch}, Batch {i}, Loss: {cost.item():.4f}")

                if i % 100 == 0 and i > 0:  # 每100个batch进行一次评估
                    with torch.no_grad():
                        self.net.eval()
                        mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                        self.net.train()

                        if mean_ap > self.best_mAP:
                            self.best_mAP = mean_ap
                            model_filename = f"{self.experiment_name.replace(':', '_').replace(' ', '_')}.pkl"
                            torch.save(self.net.state_dict(), os.path.join(self.config.model_path, model_filename))

                        print(f"[{self.experiment_name}] Epoch {epoch}, Batch {i}, mAP: {mean_ap*100:.2f}, Acc: {test_acc*100:.2f}, Best mAP: {self.best_mAP*100:.2f}")


def run_single_experiment(lambda_contrastive, lambda_gate_entropy, lambda_gate_balance, lambda_gate_feedback, args):
    """运行单个损失函数参数实验"""
    config = Config(args)
    
    # 创建实验名称
    exp_name = f"LC_{lambda_contrastive}_GE_{lambda_gate_entropy}_GB_{lambda_gate_balance}_GF_{lambda_gate_feedback}"
    
    # 修改模型保存路径以区分不同实验
    config.model_path = os.path.join(config.model_path, "comprehensive_loss_ablation", exp_name)
    os.makedirs(config.model_path, exist_ok=True)
    
    print(f"experiment name: {exp_name}")
    trainer = ComprehensiveLossAblationTrainer(config, exp_name, lambda_contrastive, lambda_gate_entropy, lambda_gate_balance, lambda_gate_feedback)
    trainer.train()


def run_comprehensive_loss_ablation(args):
    """运行综合损失函数消融实验"""
    
    # 定义对比学习权重的测试值
    contrastive_weights = [0.01, 0.05, 0.1, 0.2]
    
    # 定义门控相关权重的测试值
    gate_weights_values = [0.01, 0.05, 0.1, 0.2]

    print(f"contrastive_weights: {contrastive_weights}")
    print(f"gate_weights_values: {gate_weights_values}")
    print()
    
    # 生成所有可能的组合（为了效率，我们可以选择部分重要组合）
    # 由于完全组合数量过多，我们选择一些重要的组合
    important_combinations = []
    
    # 测试对比学习权重的主要变化
    for lc in contrastive_weights:
        important_combinations.append((lc, 0.01, 0.02, 0.20))  # 保持其他参数为基准值
    
    # 测试门控熵权重的主要变化
    for ge in gate_weights_values:
        important_combinations.append((0.10, ge, 0.02, 0.20))  # 保持其他参数为基准值
    
    # 测试门控平衡权重的主要变化
    for gb in gate_weights_values:
        important_combinations.append((0.10, 0.01, gb, 0.20))  # 保持其他参数为基准值
    
    # 测试门控反馈权重的主要变化
    for gf in gate_weights_values:
        important_combinations.append((0.10, 0.01, 0.02, gf))  # 保持其他参数为基准值
    
    # 添加一些高权重组合
    important_combinations.extend([
        (0.20, 0.10, 0.10, 0.20),
        (0.10, 0.10, 0.10, 0.20),
        (0.10, 0.05, 0.10, 0.20),
        (0.10, 0.01, 0.10, 0.20),
        (0.10, 0.01, 0.05, 0.20),
        (0.10, 0.01, 0.02, 0.10),
        (0.10, 0.01, 0.02, 0.05),
    ])
    
    print(f"total {len(important_combinations)} experiments:")
    for i, (lc, ge, gb, gf) in enumerate(important_combinations):
        print(f"combination {i+1}: LC={lc}, GE={ge}, GB={gb}, GF={gf}")
    print()
    
    for i, (lambda_contrastive, lambda_gate_entropy, lambda_gate_balance, lambda_gate_feedback) in enumerate(important_combinations):
        print(f"combination {i+1}/{len(important_combinations)}")
        print(f"  params: LC={lambda_contrastive}, GE={lambda_gate_entropy}, GB={lambda_gate_balance}, GF={lambda_gate_feedback}")
        run_single_experiment(lambda_contrastive, lambda_gate_entropy, lambda_gate_balance, lambda_gate_feedback, args)
        print(f"{i+1} completion\n")


def main():
    args = parse_args()
    run_comprehensive_loss_ablation(args)


if __name__ == '__main__':
    main()