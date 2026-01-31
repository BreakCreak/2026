"""
运行参数敏感性分析的主脚本
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

def show_analysis_options():
    """显示分析选项"""
    print("\n参数敏感性分析选项:")
    print("1. 快速参数敏感性分析 (param_sensitivity_analysis.py)")
    print("2. 详细参数分析与可视化 (detailed_param_analysis.py)")
    print("3. 运行所有分析")
    print("4. 退出")

def main():
    print("损失函数参数敏感性分析套件")
    print("此脚本将帮助您分析不同损失函数参数对训练结果的影响")
    
    while True:
        show_analysis_options()
        choice = input("\n请选择要运行的分析 (1-4): ").strip()
        
        if choice == '1':
            print("\n运行快速参数敏感性分析...")
            args = parse_args()
            cmd = f"python param_sensitivity_analysis.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 3)} --seed {args.seed} --model_path {args.model_path}"
            ret = run_command(cmd)
            print(f"快速参数分析完成，返回码: {ret}\n")
            
        elif choice == '2':
            print("\n运行详细参数分析与可视化...")
            args = parse_args()
            cmd = f"python detailed_param_analysis.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 3)} --seed {args.seed} --model_path {args.model_path}"
            ret = run_command(cmd)
            print(f"详细参数分析完成，返回码: {ret}\n")
            
        elif choice == '3':
            print("\n运行所有参数分析...")
            args = parse_args()
            
            # 运行快速分析
            cmd1 = f"python param_sensitivity_analysis.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 3)} --seed {args.seed} --model_path {args.model_path}"
            ret1 = run_command(cmd1)
            print(f"快速参数分析完成，返回码: {ret1}")
            
            # 运行详细分析
            cmd2 = f"python detailed_param_analysis.py --data_path {args.data_path} --modal {args.modal} --feature_fps {args.feature_fps} --num_segments {args.num_segments} --len_feature {args.len_feature} --num_workers {args.num_workers} --lr {args.lr} --batch_size {args.batch_size} --num_epochs {min(args.num_epochs, 3)} --seed {args.seed} --model_path {args.model_path}"
            ret2 = run_command(cmd2)
            print(f"详细参数分析完成，返回码: {ret2}")
            
            print("\n所有参数分析完成!\n")
            
        elif choice == '4':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入\n")


if __name__ == '__main__':
    main()