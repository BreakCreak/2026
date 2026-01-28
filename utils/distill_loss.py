import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    蒸馏损失函数，用于分支间的知识迁移
    """
    def __init__(self, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        """
        计算蒸馏损失
        Args:
            student_output: 学生分支的输出
            teacher_output: 教师分支的输出（通常停止梯度）
        Returns:
            蒸馏损失值
        """
        # 使用KL散度作为蒸馏损失
        loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output.detach() / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss


class AdaptiveDistillationLoss(nn.Module):
    """
    自适应蒸馏损失，根据分支性能动态调整蒸馏强度
    """
    def __init__(self, temperature=3.0):
        super(AdaptiveDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, weak_branch_output, strong_branch_output, performance_diff=None):
        """
        计算自适应蒸馏损失
        Args:
            weak_branch_output: 较弱分支的输出（学生）
            strong_branch_output: 较强分支的输出（教师，停止梯度）
            performance_diff: 性能差异（可选），用于调整蒸馏强度
        """
        # 如果没有提供性能差异，默认蒸馏强度为1.0
        if performance_diff is None:
            alpha = 1.0
        else:
            # 根据性能差异调整蒸馏强度
            alpha = torch.clamp(performance_diff, min=0.1, max=1.0)
        
        # 计算蒸馏损失
        soft_student = F.log_softmax(weak_branch_output / self.temperature, dim=1)
        soft_teacher = F.softmax(strong_branch_output.detach() / self.temperature, dim=1)
        
        distill_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        return alpha * distill_loss


class MutualLearningLoss(nn.Module):
    """
    相互学习损失，促进分支间的协同学习
    """
    def __init__(self, temperature=3.0):
        super(MutualLearningLoss, self).__init__()
        self.temperature = temperature

    def forward(self, branch1_output, branch2_output):
        """
        计算相互学习损失
        Args:
            branch1_output: 分支1的输出
            branch2_output: 分支2的输出
        """
        # 分支1学习分支2的知识
        loss_1_to_2 = F.kl_div(
            F.log_softmax(branch1_output / self.temperature, dim=1),
            F.softmax(branch2_output.detach() / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 分支2学习分支1的知识
        loss_2_to_1 = F.kl_div(
            F.log_softmax(branch2_output / self.temperature, dim=1),
            F.softmax(branch1_output.detach() / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 对称相互学习损失
        mutual_loss = 0.5 * (loss_1_to_2 + loss_2_to_1)
        
        return mutual_loss