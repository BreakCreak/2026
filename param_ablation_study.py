"""
参数消融实验脚本
此脚本用于系统地测试不同损失函数权重的影响
每次只变动一个参数，保持其他参数不变
"""

import subprocess
import os
import argparse
from itertools import product

def run_single_experiment(contrastive_weight, action_consistent_weight, gate_regularization_weight, experiment_name):
    """运行单个实验"""
    cmd = [
        "python", "main_thumos.py",
        "--contrastive_weight", str(contrastive_weight),
        "--action_consistent_weight", str(action_consistent_weight),
        "--gate_regularization_weight", str(gate_regularization_weight),
        "--experiment_name", experiment_name
    ]
    
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        print(f"Experiment {experiment_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_name} failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_name} timed out")
        return False

def run_param_ablation_study():
    """运行参数消融研究"""
    print("开始参数消融实验...")
    
    # 定义要测试的参数范围
    contrastive_weights = [0.05, 0.1, 0.2]
    action_consistent_weights = [0.05, 0.1, 0.2]
    gate_regularization_weights = [0.01, 0.05, 0.1, 0.2]
    
    # 基准参数 (0.1, 0.1, 0.01)
    baseline_contrastive = 0.1
    baseline_action_consistent = 0.1
    baseline_gate_regularization = 0.01
    
    experiments = []
    
    # 测试 adjusted_contrastive_loss 参数 (保持其他参数为基准值)
    for cw in contrastive_weights:
        exp_name = f"contrastive_{cw}_baseline_others"
        experiments.append({
            'contrastive_weight': cw,
            'action_consistent_weight': baseline_action_consistent,
            'gate_regularization_weight': baseline_gate_regularization,
            'name': exp_name
        })
    
    # 测试 action_consistent_loss 参数 (保持其他参数为基准值)
    for acw in action_consistent_weights:
        exp_name = f"action_consistent_{acw}_baseline_others"
        experiments.append({
            'contrastive_weight': baseline_contrastive,
            'action_consistent_weight': acw,
            'gate_regularization_weight': baseline_gate_regularization,
            'name': exp_name
        })
    
    # 测试 gate_regularization_loss 参数 (保持其他参数为基准值)
    for grw in gate_regularization_weights:
        exp_name = f"gate_regularization_{grw}_baseline_others"
        experiments.append({
            'contrastive_weight': baseline_contrastive,
            'action_consistent_weight': baseline_action_consistent,
            'gate_regularization_weight': grw,
            'name': exp_name
        })
    
    print(f"总共 {len(experiments)} 个实验待执行")
    
    results = []
    for exp in experiments:
        success = run_single_experiment(
            exp['contrastive_weight'],
            exp['action_consistent_weight'],
            exp['gate_regularization_weight'],
            exp['name']
        )
        results.append({'experiment': exp['name'], 'success': success})
        
        print(f"Completed: {exp['name']}")
        print("-" * 50)
    
    # 输出结果摘要
    print("\n实验结果摘要:")
    print("=" * 60)
    successful = sum(1 for r in results if r['success'])
    print(f"成功完成: {successful}/{len(results)}")
    
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{result['experiment']}: {status}")

def main():
    parser = argparse.ArgumentParser(description='参数消融实验')
    parser.add_argument('--mode', choices=['individual', 'full_combination'], 
                       default='individual', 
                       help='实验模式: individual(单参数变化) 或 full_combination(全组合)')
    args = parser.parse_args()
    
    if args.mode == 'individual':
        run_param_ablation_study()
    elif args.mode == 'full_combination':
        run_full_combination_study()

def run_full_combination_study():
    """运行完整的参数组合实验 (可选)"""
    print("开始完整参数组合实验...")
    
    contrastive_weights = [0.05, 0.1, 0.2]
    action_consistent_weights = [0.05, 0.1, 0.2]
    gate_regularization_weights = [0.01, 0.05, 0.1, 0.2]
    
    experiments = []
    
    # 生成所有参数组合
    for cw, acw, grw in product(contrastive_weights, action_consistent_weights, gate_regularization_weights):
        exp_name = f"full_combo_c{cw}_ac{acw}_gr{grw}"
        experiments.append({
            'contrastive_weight': cw,
            'action_consistent_weight': acw,
            'gate_regularization_weight': grw,
            'name': exp_name
        })
    
    print(f"总共 {len(experiments)} 个完整组合实验待执行")
    
    results = []
    for exp in experiments:
        success = run_single_experiment(
            exp['contrastive_weight'],
            exp['action_consistent_weight'],
            exp['gate_regularization_weight'],
            exp['name']
        )
        results.append({'experiment': exp['name'], 'success': success})
        
        print(f"Completed: {exp['name']}")
        print("-" * 50)
    
    # 输出结果摘要
    print("\n完整组合实验结果摘要:")
    print("=" * 60)
    successful = sum(1 for r in results if r['success'])
    print(f"成功完成: {successful}/{len(results)}")

if __name__ == "__main__":
    main()