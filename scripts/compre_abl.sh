#!/bin/bash

echo "开始运行参数消融实验..."
echo "==============================="

echo ""
echo "1. 测试 adjusted_contrastive_loss 参数 (保持 action_consistent=0.1, gate_reg=0.01)"
echo "-------------------------------"
python main_thumos.py --contrastive_weight 0.05 --action_consistent_weight 0.1 --gate_regularization_weight 0.01 --exp_name "abl_contrastive_0.05" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.1 --gate_regularization_weight 0.01 --exp_name "abl_contrastive_0.1" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.2 --action_consistent_weight 0.1 --gate_regularization_weight 0.01 --exp_name "abl_contrastive_0.2" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

echo ""
echo "2. 测试 action_consistent_loss 参数 (保持 contrastive=0.1, gate_reg=0.01)"
echo "-------------------------------"
python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.05 --gate_regularization_weight 0.01 --exp_name "abl_action_consistent_0.05" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.1 --gate_regularization_weight 0.01 --exp_name "abl_action_consistent_0.1" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.2 --gate_regularization_weight 0.01 --exp_name "abl_action_consistent_0.2" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

echo ""
echo "3. 测试 gate_regularization_loss 参数 (保持 contrastive=0.1, action_consistent=0.1)"
echo "-------------------------------"
python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.1 --gate_regularization_weight 0.01 --exp_name "abl_gate_regularization_0.01" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.1 --gate_regularization_weight 0.05 --exp_name "abl_gate_regularization_0.05" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.1 --gate_regularization_weight 0.1 --exp_name "abl_gate_regularization_0.1" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

python main_thumos.py --contrastive_weight 0.1 --action_consistent_weight 0.1 --gate_regularization_weight 0.2 --exp_name "abl_gate_regularization_0.2" --model_name ThumosModel --num_epochs 500 --detection_inf_step 50 --soft_nms --data_path '../THUMOS14'
echo ""

echo "所有参数消融实验完成！"