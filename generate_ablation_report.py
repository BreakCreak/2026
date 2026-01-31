"""
生成消融实验报告的脚本
该脚本会分析实验结果并生成汇总报告
"""

import os
import re
from datetime import datetime

def analyze_training_logs():
    """分析训练日志文件"""
    log_files = []
    for file in os.listdir("."):
        if file.startswith("training_log") and file.endswith(".txt"):
            log_files.append(file)
    
    if not log_files:
        print("未找到训练日志文件。运行实验后会生成日志文件。")
        return
    
    print("发现以下日志文件:")
    for log_file in log_files:
        print(f"- {log_file}")
    
    # 这里可以添加解析日志文件的逻辑
    print("\n提示: 可以扩展此脚本来自动解析训练日志中的最佳mAP等指标")

def generate_experiment_plan():
    """生成实验计划报告"""
    print("参数消融实验计划")
    print("=" * 50)
    print("基准参数: contrastive_weight=0.1, action_consistent_weight=0.1, gate_regularization_weight=0.01")
    print()
    
    # 实验参数
    contrastive_weights = [0.05, 0.1, 0.2]
    action_consistent_weights = [0.05, 0.1, 0.2]
    gate_regularization_weights = [0.01, 0.05, 0.1, 0.2]
    
    print("1. 测试 contrastive_weight 参数:")
    for cw in contrastive_weights:
        print(f"   - contrastive_weight={cw}, action_consistent_weight=0.1, gate_regularization_weight=0.01")
    print(f"   总计: {len(contrastive_weights)} 个实验")
    print()
    
    print("2. 测试 action_consistent_weight 参数:")
    for acw in action_consistent_weights:
        print(f"   - contrastive_weight=0.1, action_consistent_weight={acw}, gate_regularization_weight=0.01")
    print(f"   总计: {len(action_consistent_weights)} 个实验")
    print()
    
    print("3. 测试 gate_regularization_weight 参数:")
    for grw in gate_regularization_weights:
        print(f"   - contrastive_weight=0.1, action_consistent_weight=0.1, gate_regularization_weight={grw}")
    print(f"   总计: {len(gate_regularization_weights)} 个实验")
    print()
    
    total_experiments = len(contrastive_weights) + len(action_consistent_weights) + len(gate_regularization_weights)
    print(f"总计实验数量: {total_experiments}")
    
    print("\n运行方式:")
    print("- 使用 simple_param_ablation.py 脚本自动运行所有实验")
    print("- 或使用 run_ablation_experiments.bat 批处理文件运行")

def main():
    print("消融实验报告生成器")
    print("=" * 50)
    
    generate_experiment_plan()
    print()
    analyze_training_logs()
    
    print("\n实验执行建议:")
    print("1. 首先运行基准实验 (所有参数为默认值) 来建立基线性能")
    print("2. 然后分别运行各参数的变化实验")
    print("3. 记录每个实验的最佳mAP和其他关键指标")
    print("4. 对比分析各参数对模型性能的影响")

if __name__ == "__main__":
    main()