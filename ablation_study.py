"""
消融实验主程序
包含所有消融实验变体的训练和评估
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

# 导入各种消融实验模型
from models.model import AICL  # E5: 完整模型
from models.e1_rgb_only_model import E1_RGB_Only
from models.e1_flow_only_model import E1_Flow_Only
from models.e1_joint_only_model import E1_Joint_Only
from models.e2_no_mixed_expert_model import E2_No_MixedExpert
from models.e3_mixed_expert_no_gate_model import E3_MixedExpert_NoGate
from models.e4_gate_off_model import E4_GateOff
from models.e6_no_gate_regularization_model import E6_NoGateRegularization

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})
np.set_printoptions(threshold=np.inf)

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


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


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):                #　　0.1
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
    """
    门控熵正则化：防止门控权重退化为常数，鼓励更多样化的分支选择
    Args:
        gate_weights: [B, 2, T] (现在是2个分支)
    """
    p = gate_weights + 1e-6  # 防止log(0)
    entropy = -torch.sum(p * torch.log(p), dim=1)  # [B, T]
    return entropy.mean()


def branch_balance_loss(gate_weights):
    """
    分支平衡损失：防止某个分支被永久忽略
    Args:
        gate_weights: [B, 2, T]
    """
    # 计算每个分支在整个序列上的平均激活值
    avg_activation = torch.mean(gate_weights, dim=2)  # [B, 2]
    
    # 计算批次级别平均激活值
    batch_avg = torch.mean(avg_activation, dim=0)  # [2]
    
    # 希望两个分支都被激活（理想情况下每个分支平均激活率为0.5）
    ideal_ratio = 0.5
    balance_loss = torch.mean((batch_avg - ideal_ratio) ** 2)
    
    return balance_loss


def expert_diversity_loss(e1, e2):
    """
    专家多样性损失，防止两个混合专家学习到相同的表示
    Args:
        e1: 第一个专家的输出 [B, T, C]
        e2: 第二个专家的输出 [B, T, C]
    """
    # 将张量转为 [B, C, T] 以便进行归一化
    e1 = e1.permute(0, 2, 1)
    e2 = e2.permute(0, 2, 1)
    
    # 对通道维度进行归一化
    e1 = F.normalize(e1.mean(dim=1), dim=1)  # [B, T]
    e2 = F.normalize(e2.mean(dim=1), dim=1)  # [B, T]
    
    return (e1 * e2).sum(dim=1).mean()


class AblationTrainer():
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
        self.best_mAP = -1 # init
        self.step = 0
        self.total_loss_per_epoch = 0

        # 为不同的实验类型设置不同的损失权重
        if experiment_type in ["E1_RGB", "E1_Flow", "E1_Joint"]:
            # Level 1 实验不需要门控相关损失
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.0
            self.lambda_gate_balance = 0.0
            self.lambda_gate_feedback = 0.0
        elif experiment_type == "E4_GateOff":
            # E4实验不使用门控反馈损失
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.01
            self.lambda_gate_balance = 0.02
            self.lambda_gate_feedback = 0.0
        elif experiment_type in ["E5_Full", "E6_NoReg"]:
            # 完整模型和无正则化模型
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.01 if experiment_type == "E5_Full" else 0.0
            self.lambda_gate_balance = 0.02 if experiment_type == "E5_Full" else 0.0
            self.lambda_gate_feedback = 0.05 if experiment_type == "E5_Full" else 0.0
        else:
            # 其他实验的默认权重
            self.lambda_contrastive = 0.01
            self.lambda_gate_entropy = 0.01
            self.lambda_gate_balance = 0.02
            self.lambda_gate_feedback = 0.05

    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = f"{self.experiment_type}_CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print(f"[{self.experiment_type}] cls_acc={test_acc*100:.5f} map={_mean_ap*100:.5f}")


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments

    def calculate_gate_feedback_loss(self, gate_weights, action_branch1, action_branch2, topk_indices):
        """
        计算门控反馈损失，让门控能够感知到选择的准确性
        如果门控选择了某个分支，而该分支实际表现更好，则给予正反馈
        如果门控选择了某个分支，但另一个分支表现更好，则给予负反馈
        """
        # 获取top-k区域的门控权重和分支actionness
        # gate_weights: [B, 2, T] -> [B, 2, top_k]
        batch_size = gate_weights.size(0)
        
        # 确保topk_indices维度与action_branch1和action_branch2匹配
        # 检查topk_indices的形状，如果需要调整
        if len(topk_indices.shape) != 2 or topk_indices.shape[0] != action_branch1.shape[0]:
            # 如果topk_indices是其他形状，需要重新获取
            _, temp_topk_indices = torch.topk(action_branch1, min(10, action_branch1.size(1)), dim=1)
            topk_indices = temp_topk_indices
        
        # 使用gather提取top-k区域的分支actionness
        action_branch1_topk = torch.gather(action_branch1, 1, topk_indices)  # [B, top_k]
        action_branch2_topk = torch.gather(action_branch2, 1, topk_indices)  # [B, top_k]
        
        # 计算每个分支在top-k区域的平均actionness（代表分支质量）
        branch1_quality = action_branch1_topk.mean(dim=1, keepdim=True)  # [B, 1]
        branch2_quality = action_branch2_topk.mean(dim=1, keepdim=True)  # [B, 1]
        
        # 比较两个分支的质量
        # 如果branch1_quality > branch2_quality，则branch1相对优势为1，branch2为0，反之亦然
        mask_branch1_better = branch1_quality > branch2_quality  # [B, 1]
        
        # 创建相对优势张量 [B, 2, 1]
        relative_advantage = torch.zeros((batch_size, 2, 1), device=gate_weights.device)
        relative_advantage[:, 0, :] = mask_branch1_better.float()  # branch1的优势 [B, 1]
        relative_advantage[:, 1, :] = (~mask_branch1_better).float()  # branch2的优势 [B, 1]
        
        # 扩展topk_indices到门控权重维度
        expanded_indices = topk_indices.unsqueeze(1).expand(-1, 2, -1)  # [B, 2, top_k]
        
        # 提取top-k区域的门控权重
        gate_weights_topk = torch.gather(gate_weights, 2, expanded_indices)  # [B, 2, top_k]
        
        # 计算门控权重与理想选择之间的差距
        # 计算top-k区域的平均门控权重 [B, 2, 1]
        avg_gate_weights = gate_weights_topk.mean(dim=2, keepdim=True)  # [B, 2, 1]
        
        # 计算门控反馈损失：门控权重与理想选择越接近，损失越小
        gate_feedback_loss = F.mse_loss(avg_gate_weights, relative_advantage)
        
        return gate_feedback_loss

    def calculate_all_losses(self, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, cas_top, label, topk_indices, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2):
        self.contrastive_criterion = ContrastiveLoss()
        
        # 原有的对比损失
        L_c = self.contrastive_criterion(contrast_pairs)
        L_r = self.contrastive_criterion(contrast_pairs_r)
        L_f = self.contrastive_criterion(contrast_pairs_f)
        
        # 混合分支对比损失
        L_m = self.contrastive_criterion(contrast_pairs_m)
        L_m2 = self.contrastive_criterion(contrast_pairs_m2)
        
        # RGB+Flow相加分支对比损失
        L_b1 = self.contrastive_criterion(contrast_pairs_b1)
        L_b1_2 = self.contrastive_criterion(contrast_pairs_b1_2)
        
        # 分支独立对比损失（用于门控反馈）
        L_b1_ind = self.contrastive_criterion(contrast_pairs_b1_ind)
        L_m_ind = self.contrastive_criterion(contrast_pairs_m_ind)
        
        # 总对比损失
        loss_contrastive = L_c + L_r + L_f + 0.5 * L_m + 0.3 * L_m2 + 0.5 * L_b1 + 0.3 * L_b1_2 + 0.4 * L_b1_ind + 0.4 * L_m_ind

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = 0.5 * F.mse_loss(action_flow, action_rgb) + 0.5 * F.mse_loss(action_rgb, action_flow)
        action_consistent_loss = 0.5 * F.mse_loss(actionness1, actionness2) + 0.5 * F.mse_loss(actionness2, actionness1)
    
        # 计算门控熵损失（根据实验类型决定是否使用）
        gate_ent_loss = gate_entropy_loss(gate_weights) if self.lambda_gate_entropy > 0 else torch.tensor(0.0, device=gate_weights.device)
    
        # 计算分支平衡损失（根据实验类型决定是否使用）
        gate_balance_loss = branch_balance_loss(gate_weights) if self.lambda_gate_balance > 0 else torch.tensor(0.0, device=gate_weights.device)
    
        # 增强的门控反馈机制（根据实验类型决定是否使用）
        gate_feedback_loss = self.calculate_gate_feedback_loss(gate_weights, action_branch1, action_branch2, topk_indices) if self.lambda_gate_feedback > 0 else torch.tensor(0.0, device=gate_weights.device)

        # 根据实验类型调整损失权重
        cost = (base_loss + 
                class_agnostic_loss + 
                5*modality_consistent_loss + 
                self.lambda_contrastive*loss_contrastive + 
                0.1*action_consistent_loss + 
                self.lambda_gate_entropy * gate_ent_loss + 
                self.lambda_gate_balance * gate_balance_loss + 
                self.lambda_gate_feedback * gate_feedback_loss)

        return cost

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                model_filename = f"{self.experiment_type}_CAS_Only.pkl"
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, model_filename))

            print(f"[{self.experiment_type}] epoch={epoch:5d}  step={self.step:5d}  Loss={self.total_loss_per_epoch:.4f}  cls_acc={test_acc * 100:5.2f}  best_map={self.best_mAP * 100:5.2f}")

            self.total_loss_per_epoch = 0

    def forward_pass(self, _data):
        cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2 = self.net(_data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.unsqueeze(2).detach(),
                                                              action_rgb.unsqueeze(2))


        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        # _, topk_indices1 = torch.topk(combined_cas, r, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2

    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):

            for _data, _label, temp_anno, _, _ in self.train_loader:

                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                # forward pass
                cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2 = self.forward_pass(_data)

                # calcualte pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)

                # losses
                cost = self.calculate_all_losses(contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, cas_top, _label, topk_indices, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2)

                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)


def run_ablation_experiment(experiment_type, args):
    """运行指定的消融实验"""
    config = Config(args)
    set_seed(config)
    
    # 修改模型保存路径以区分不同实验
    config.model_path = os.path.join(config.model_path, experiment_type)
    os.makedirs(config.model_path, exist_ok=True)
    
    trainer = AblationTrainer(config, experiment_type)
    
    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


def main():
    args = parse_args()
    
    # 定义所有消融实验
    experiments = [
        # "E1_RGB",      # E1-1: RGB only
        # "E1_Flow",     # E1-2: Flow only
        # "E1_Joint",    # E1-3: Joint only
        "E2_NoME",     # E2: 无 Mixed Expert
        "E3_ME_NoGate", # E3: Mixed Expert 无 Gate
        "E4_GateOff",   # E4: Gate-off
        "E5_Full",      # E5: 完整模型
        "E6_NoReg"      # E6: 无门控正则化
    ]
    
    print("ablation...")
    print("list:")
    # print("- E1-1: RGB only")
    # print("- E1-2: Flow only")
    # print("- E1-3: Joint only")
    print("- E2: no Mixed Expert")
    print("- E3: Mixed Expert no Gate")
    print("- E4: Gate-off")
    print("- E5: completed (E5)")
    print("- E6: no gate")
    print()
    
    for exp in experiments:
        print(f"running experiment: {exp}")
        run_ablation_experiment(exp, args)
        print(f"experiment: {exp} completed\n")


if __name__ == '__main__':
    main()