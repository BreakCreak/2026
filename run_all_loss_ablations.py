"""
运行所有损失函数消融实验的主脚本
包含混合专家和门控机制的损失函数参数测试
"""

import subprocess
import sys
import os
from config.config_thumos import parse_args

def run_command(command):
    """运行命令并打印输出"""
    print(f"运行命令: {command}")
    result = subprocess.run(command, shell=True)
    return result.returncode

def run_loss_ablation_experiments():
    """运行所有损失函数消融实验"""
    
    print("="*60)
    print("开始运行所有损失函数消融实验")
    print("="*60)
    
    # 获取命令行参数
    args = parse_args()
    
    # 构建基本命令
    base_cmd = f"python "
    
    print("\n1. 运行快速门控损失函数消融实验...")
    cmd1 = f"python quick_loss_ablation.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 5)} --seed {args.seed} --model_path {args.model_path}"
    ret1 = run_command(cmd1)
    print(f"快速门控损失实验完成，返回码: {ret1}")
    
    print("\n2. 运行混合专家损失函数消融实验...")
    cmd2 = f"python expert_loss_ablation.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 3)} --seed {args.seed} --model_path {args.model_path}"
    ret2 = run_command(cmd2)
    print(f"混合专家损失实验完成，返回码: {ret2}")
    
    print("\n3. 运行综合损失函数消融实验...")
    cmd3 = f"python comprehensive_loss_ablation.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 2)} --seed {args.seed} --model_path {args.model_path}"
    ret3 = run_command(cmd3)
    print(f"综合损失实验完成，返回码: {ret3}")
    
    print("\n" + "="*60)
    print("所有损失函数消融实验运行完成!")
    print("实验结果保存在各自对应的子目录中")
    print("="*60)


def show_experiment_summary():
    """显示实验摘要"""
    print("\n损失函数消融实验摘要:")
    print("- 快速门控损失实验: 测试门控相关权重的不同组合 (GE, GB, GF)")
    print("- 混合专家损失实验: 测试是否包含混合专家对比损失的影响")
    print("- 综合损失实验: 测试多种权重参数的组合效果")
    print("\n权重参数说明:")
    print("- LC: 对比学习权重 (0.01, 0.05, 0.1, 0.2)")
    print("- GE: 门控熵权重 (0.01, 0.05, 0.1, 0.2)")
    print("- GB: 门控平衡权重 (0.01, 0.05, 0.1, 0.2)")
    print("- GF: 门控反馈权重 (0.01, 0.05, 0.1, 0.2)")


def main():
    print("损失函数参数消融实验套件")
    print("此脚本将运行所有相关的损失函数消融实验")
    
    show_experiment_summary()
    
    response = input("\n是否开始运行所有实验? (y/n): ")
    if response.lower() in ['y', 'yes']:
        run_loss_ablation_experiments()
    else:
        print("取消运行实验")


if __name__ == '__main__':
    main()