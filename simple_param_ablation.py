"""
Simple Parameter Ablation Study Script
This script systematically tests the impact of different loss function weights.
Each time, only one parameter is varied while keeping others at baseline values (0.1, 0.1, 0.01).
"""

import subprocess
import sys
import os
import argparse

def run_experiment(contrastive_weight, action_consistent_weight, gate_regularization_weight, output_suffix):
    """Run a single experiment"""
    cmd = [
        sys.executable, "main_thumos.py",
        "--contrastive_weight", str(contrastive_weight),
        "--action_consistent_weight", str(action_consistent_weight),
        "--gate_regularization_weight", str(gate_regularization_weight)
    ]
    
    print(f"Running experiment: contrastive={contrastive_weight}, action_consistent={action_consistent_weight}, gate_reg={gate_regularization_weight}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
        print(f"Experiment {output_suffix} executed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment {output_suffix} failed! Error: {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Experiment {output_suffix} timed out!")
        return False

def main():
    print("Starting parameter ablation study...")
    print("Baseline parameters: contrastive=0.1, action_consistent=0.1, gate_regularization=0.01")
    print("-" * 70)
    
    # 定义各参数的测试值
    contrastive_weights = [0.05, 0.1, 0.2]
    action_consistent_weights = [0.05, 0.1, 0.2]
    gate_regularization_weights = [0.01, 0.05, 0.1, 0.2]
    
    # 基准参数
    baseline_contrastive = 0.1
    baseline_action_consistent = 0.1
    baseline_gate_regularization = 0.01
    
    total_experiments = len(contrastive_weights) + len(action_consistent_weights) + len(gate_regularization_weights)
    print(f"Total {total_experiments} experiments will be run")
    print("-" * 70)
    
    experiment_count = 0
    
    # 1. Test contrastive_weight parameter (keeping other parameters at baseline)
    print("1. Testing adjusted_contrastive_loss parameter...")
    for cw in contrastive_weights:
        experiment_count += 1
        print(f"\n[{experiment_count}/{total_experiments}] ", end="")
        success = run_experiment(
            contrastive_weight=cw,
            action_consistent_weight=baseline_action_consistent,
            gate_regularization_weight=baseline_gate_regularization,
            output_suffix=f"contrastive_{cw}"
        )
        if not success:
            print(f"Experiment contrastive_{cw} failed, continuing to next...")
    
    # 2. Test action_consistent_weight parameter (keeping other parameters at baseline)
    print("\n2. Testing action_consistent_loss parameter...")
    for acw in action_consistent_weights:
        experiment_count += 1
        print(f"\n[{experiment_count}/{total_experiments}] ", end="")
        success = run_experiment(
            contrastive_weight=baseline_contrastive,
            action_consistent_weight=acw,
            gate_regularization_weight=baseline_gate_regularization,
            output_suffix=f"action_consistent_{acw}"
        )
        if not success:
            print(f"Experiment action_consistent_{acw} failed, continuing to next...")
    
    # 3. Test gate_regularization_weight parameter (keeping other parameters at baseline)
    print("\n3. Testing gate_regularization_loss parameter...")
    for grw in gate_regularization_weights:
        experiment_count += 1
        print(f"\n[{experiment_count}/{total_experiments}] ", end="")
        success = run_experiment(
            contrastive_weight=baseline_contrastive,
            action_consistent_weight=baseline_action_consistent,
            gate_regularization_weight=grw,
            output_suffix=f"gate_regularization_{grw}"
        )
        if not success:
            print(f"Experiment gate_regularization_{grw} failed, continuing to next...")
    
    print("\n" + "="*70)
    print("All parameter ablation experiments completed!")
    print("Summary:")
    print(f"- contrastive_weight tests: {len(contrastive_weights)} values")
    print(f"- action_consistent_weight tests: {len(action_consistent_weights)} values") 
    print(f"- gate_regularization_weight tests: {len(gate_regularization_weights)} values")
    print(f"- Total experiments executed: {total_experiments}")

if __name__ == "__main__":
    main()