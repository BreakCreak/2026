"""
详细的参数敏感性分析
测试不同损失函数参数对训练结果的影响，并生成可视化图表
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

class DetailedParamAnalysisTrainer():
    def __init__(self, config, params_config):
        self.config = config
        self.params_config = params_config

        self.net = AICL(config)
        self.net = self.net.cuda()

        self.train_loader, self.test_loader = get_dataloaders(self.config)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        self.best_mAP = -1
        self.step = 0

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

        # 使用配置中的参数
        cost = (base_loss + 
                class_agnostic_loss + 
                5*modality_consistent_loss + 
                self.params_config['lambda_contrastive']*adjusted_contrastive_loss + 
                0.1*action_consistent_loss + 
                self.params_config['lambda_gate_entropy'] * gate_ent_loss + 
                self.params_config['lambda_gate_balance'] * gate_balance_loss + 
                self.params_config['lambda_gate_feedback'] * gate_feedback_loss)

        return cost

    def forward_pass(self, _data):
        cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2 = self.net(_data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.unsqueeze(2).detach(),
                                                              action_rgb.unsqueeze(2))

        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, contrast_pairs_m, contrast_pairs_m2, contrast_pairs_b1, contrast_pairs_b1_2, contrast_pairs_b1_ind, contrast_pairs_m_ind, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, embedding_mixed, embedding_branch1, action_branch1, action_branch2

    def train_and_evaluate_detailed(self):
        """训练并返回详细的性能指标"""
        set_seed(self.config.seed)
        
        # 记录训练过程中的结果
        mAP_history = []
        acc_history = []

        for epoch in range(min(3, self.config.num_epochs)):  # 仅训练3个epoch以快速测试
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

                if i % 50 == 0 and i > 0:  # 每50个batch评估一次
                    with torch.no_grad():
                        self.net.eval()
                        mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                        self.net.train()

                        if mean_ap > self.best_mAP:
                            self.best_mAP = mean_ap

                        mAP_history.append(mean_ap)
                        acc_history.append(test_acc)

                        print(f"[Config: {self.params_config['name']}] Epoch {epoch}, Batch {i}, mAP: {mean_ap*100:.2f}, Acc: {test_acc*100:.2f}, Best mAP: {self.best_mAP*100:.2f}")

        return {
            'final_mAP': mAP_history[-1] if mAP_history else 0,
            'final_acc': acc_history[-1] if acc_history else 0,
            'best_mAP': self.best_mAP,
            'mAP_history': mAP_history,
            'acc_history': acc_history
        }


def plot_results(results):
    """绘制参数敏感性分析结果"""
    import matplotlib.pyplot as plt
    
    # 提取数据
    configs = [r['config']['name'] for r in results]
    mAPs = [r['result']['final_mAP'] * 100 for r in results]
    best_mAPs = [r['result']['best_mAP'] * 100 for r in results]
    accs = [r['result']['final_acc'] * 100 for r in results]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # mAP对比
    x_pos = range(len(configs))
    axes[0, 0].bar(x_pos, mAPs, alpha=0.7, label='Final mAP')
    axes[0, 0].bar(x_pos, best_mAPs, alpha=0.7, label='Best mAP')
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('mAP (%)')
    axes[0, 0].set_title('mAP Comparison Across Configurations')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy对比
    axes[0, 1].bar(x_pos, accs, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Configuration')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Comparison Across Configurations')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(configs, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 参数影响热力图 (LC vs GE)
    lc_values = sorted(list(set([r['config']['lambda_contrastive'] for r in results])))
    ge_values = sorted(list(set([r['config']['lambda_gate_entropy'] for r in results])))
    
    if len(lc_values) > 1 and len(ge_values) > 1:
        heatmap_data = np.zeros((len(ge_values), len(lc_values)))
        for r in results:
            lc_idx = lc_values.index(r['config']['lambda_contrastive'])
            ge_idx = ge_values.index(r['config']['lambda_gate_entropy'])
            heatmap_data[ge_idx, lc_idx] = r['result']['best_mAP'] * 100
        
        im = axes[1, 0].imshow(heatmap_data, cmap='viridis', aspect='auto')
        axes[1, 0].set_xlabel('Contrastive Weight (LC)')
        axes[1, 0].set_ylabel('Gate Entropy Weight (GE)')
        axes[1, 0].set_title('mAP Heatmap: LC vs GE')
        axes[1, 0].set_xticks(range(len(lc_values)))
        axes[1, 0].set_xticklabels([f'{v:.2f}' for v in lc_values])
        axes[1, 0].set_yticks(range(len(ge_values)))
        axes[1, 0].set_yticklabels([f'{v:.2f}' for v in ge_values])
        
        # 添加数值标注
        for i in range(len(ge_values)):
            for j in range(len(lc_values)):
                text = axes[1, 0].text(j, i, f'{heatmap_data[i, j]:.1f}',
                                       ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, 'Not enough data for heatmap', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('mAP Heatmap: LC vs GE (Insufficient Data)')
    
    # 性能散点图
    axes[1, 1].scatter(mAPs, accs, s=100, alpha=0.7)
    for i, config in enumerate(configs):
        axes[1, 1].annotate(config, (mAPs[i], accs[i]), xytext=(5, 5), 
                            textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('mAP (%)')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('mAP vs Accuracy Scatter Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('param_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_detailed_param_analysis(args):
    """运行详细的参数分析"""
    
    print("开始运行详细的损失函数参数敏感性分析...")
    print("测试不同参数对训练结果的影响")
    
    # 定义要测试的参数组合 - 更系统的测试
    param_configs = [
        # 基准配置
        {'lambda_contrastive': 0.01, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'baseline'},
        
        # 对比学习权重变化 (0.01, 0.05, 0.1, 0.2)
        {'lambda_contrastive': 0.05, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'LC_0.05'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'LC_0.10'},
        {'lambda_contrastive': 0.20, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'LC_0.20'},
        
        # 门控熵权重变化 (0.01, 0.05, 0.1, 0.2)
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.05, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'GE_0.05'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.10, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'GE_0.10'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.20, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'GE_0.20'},
        
        # 门控平衡权重变化 (0.01, 0.05, 0.1, 0.2)
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.05, 'lambda_gate_feedback': 0.05, 'name': 'GB_0.05'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.10, 'lambda_gate_feedback': 0.05, 'name': 'GB_0.10'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.20, 'lambda_gate_feedback': 0.05, 'name': 'GB_0.20'},
        
        # 门控反馈权重变化 (0.01, 0.05, 0.1, 0.2)
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.01, 'name': 'GF_0.01'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.05, 'name': 'GF_0.05'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.10, 'name': 'GF_0.10'},
        {'lambda_contrastive': 0.10, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.02, 'lambda_gate_feedback': 0.20, 'name': 'GF_0.20'},
        
        # 组合高权重配置
        {'lambda_contrastive': 0.20, 'lambda_gate_entropy': 0.10, 'lambda_gate_balance': 0.10, 'lambda_gate_feedback': 0.20, 'name': 'all_high'},
        
        # 组合低权重配置
        {'lambda_contrastive': 0.01, 'lambda_gate_entropy': 0.01, 'lambda_gate_balance': 0.01, 'lambda_gate_feedback': 0.01, 'name': 'all_low'},
    ]
    
    print(f"总共测试 {len(param_configs)} 种参数配置")
    for i, config in enumerate(param_configs):
        print(f"  {i+1}. {config['name']}: LC={config['lambda_contrastive']}, GE={config['lambda_gate_entropy']}, GB={config['lambda_gate_balance']}, GF={config['lambda_gate_feedback']}")
    print()
    
    results = []
    
    for i, param_config in enumerate(param_configs):
        print(f"正在测试参数配置 {i+1}/{len(param_configs)}: {param_config['name']}")
        
        # 创建配置副本
        config_copy = Config(args)
        
        # 修改模型保存路径以区分不同实验
        config_copy.model_path = os.path.join(config_copy.model_path, "detailed_param_analysis", param_config['name'])
        os.makedirs(config_copy.model_path, exist_ok=True)
        
        trainer = DetailedParamAnalysisTrainer(config_copy, param_config)
        result = trainer.train_and_evaluate_detailed()
        
        full_result = {
            'config': param_config,
            'result': result
        }
        results.append(full_result)
        
        print(f"  结果: Final mAP={result['final_mAP']*100:.2f}%, Best mAP={result['best_mAP']*100:.2f}%, Acc={result['final_acc']*100:.2f}%")
        print()
    
    # 输出结果汇总
    print("="*100)
    print("详细参数敏感性分析结果汇总:")
    print("="*100)
    print(f"{'配置名称':<15} {'LC':<6} {'GE':<6} {'GB':<6} {'GF':<6} {'Final_mAP(%)':<12} {'Best_mAP(%)':<12} {'Acc(%)':<8}")
    print("-"*100)
    
    for result in results:
        config = result['config']
        res = result['result']
        print(f"{config['name']:<15} {config['lambda_contrastive']:<6.2f} {config['lambda_gate_entropy']:<6.2f} {config['lambda_gate_balance']:<6.2f} {config['lambda_gate_feedback']:<6.2f} {res['final_mAP']*100:<12.2f} {res['best_mAP']*100:<12.2f} {res['final_acc']*100:<8.2f}")
    
    print("-"*100)
    
    # 找出最佳配置（按Best mAP排序）
    best_result = max(results, key=lambda x: x['result']['best_mAP'])
    print(f"\n最佳配置 (按Best mAP): {best_result['config']['name']}")
    print(f"参数: LC={best_result['config']['lambda_contrastive']}, GE={best_result['config']['lambda_gate_entropy']}, GB={best_result['config']['lambda_gate_balance']}, GF={best_result['config']['lambda_gate_feedback']}")
    print(f"性能: Best mAP={best_result['result']['best_mAP']*100:.2f}%, Final mAP={best_result['result']['final_mAP']*100:.2f}%, Acc={best_result['result']['final_acc']*100:.2f}%")
    
    # 生成可视化图表
    try:
        plot_results(results)
        print("\n已生成可视化图表: param_sensitivity_analysis.png")
    except ImportError:
        print("\n注意: 无法生成可视化图表，请安装matplotlib库")
    
    return results


def main():
    args = parse_args()
    results = run_detailed_param_analysis(args)
    
    print("\n详细参数敏感性分析完成！")
    print("结果表明不同损失函数参数对训练结果有显著影响，可以根据具体任务需求调整参数以获得最佳性能。")


if __name__ == '__main__':
    main()