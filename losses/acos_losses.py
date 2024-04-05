# -*- coding: utf-8 -*-
# @Time    : 2022/11/6 21:08
# @Author  : codewen77
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from losses.SupCL import SupConLoss


def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor


def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end):
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)

    weight = torch.tensor([1, 3]).float().cuda()

    loss_start = F.cross_entropy(pred_start, gold_start.long(), reduction='sum', weight=weight, ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), reduction='sum', weight=weight, ignore_index=-1)

    return 0.5 * loss_start + 0.5 * loss_end


def calculate_category_loss(pred_category, gold_category):
    return F.cross_entropy(pred_category, gold_category.long(), reduction='sum', ignore_index=-1)


def calculate_sentiment_loss(pred_sentiment, gold_sentiment):
    return F.cross_entropy(pred_sentiment, gold_sentiment.long(), reduction='sum', ignore_index=-1)


def calculate_SCL_loss(gold, pred_scores):
    SCL = SupConLoss(contrast_mode='all', temperature=0.9)  # 0.9

    answer = gold
    idxs = torch.nonzero(answer != -1).squeeze()

    answers, score_list = [], []
    for i in idxs:
        answers.append(answer[i])
        score_list.append(pred_scores[i])

    # label 维度变为:[trans_dim]
    answers = torch.stack(answers)
    # 进行维度重构 category维度:[category_nums, 1, trans_dim]
    scores = torch.stack(score_list)

    scores = F.softmax(scores, dim=1)
    scores = scores.unsqueeze(1)

    scl_loss = SCL(scores, answers)
    scl_loss /= len(scores)

    return scl_loss


class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    """

    def __init__(self, gamma=1, weight=None, ignore_index=-1):
        super(FocalLoss, self).__init__()
        # alpha
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, gold):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = F.log_softmax(pred, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = F.nll_loss(log_pt, gold, self.weight, ignore_index=self.ignore_index)
        return loss


class LDAMLoss(nn.Module):
    """
    标签分布感知边距（LDAM）损失
    作者建议对少数类引入比多数类更强的正则化，以减少它们的泛化误差。如此一来，损失函数保持了模型学习多数类并强调少数类的能力。
    它基于类别分布的信息，调整了样本的权重，从而更有效地训练模型。

    Rest 15 train_dataset:
    {
        'RESTAURANT#GENERAL': 194, 'SERVICE#GENERAL': 240, 'FOOD#GENERAL': 1, 'FOOD#QUALITY': 501,
        'FOOD#STYLE_OPTIONS': 66, 'DRINKS#STYLE_OPTIONS': 21, 'DRINKS#PRICES': 11, 'AMBIENCE#GENERAL': 157,
        'RESTAURANT#PRICES': 37, 'FOOD#PRICES': 41, 'RESTAURANT#MISCELLANEOUS': 40, 'DRINKS#QUALITY': 27,
        'LOCATION#GENERAL': 16
    }
    {
        'negative': 314, 'neutral': 34, 'positive': 1004
    }
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        """
        :param cls_num_list: 每个类别的样本数量,如[1000, 2000, 3000, 4000]
        :param max_m: max_m是LDAM Loss的超参数，控制样本权重的缩放
        :param weight: weight参数可以用于指定每个类别的样本权重，用于进一步调整类别权重
        :param s: s是LDAM Loss的另一个超参数，用于控制损失的缩放
        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # nj的四次开方
        m_list = m_list * (max_m / np.max(m_list))  # 常系数 C
        m_list = torch.cuda.FloatTensor(m_list)  # 转成 tensor
        self.m_list = m_list
        assert s > 0
        self.s = s  # 这个参数的作用论文里提过么？
        self.weight = weight  # 和频率相关的 re-weight

    def forward(self, pred, target):
        # pred: [N, C]
        # target: [N]
        idxs = torch.nonzero(target != -1).squeeze()
        answers, score_list = [], []
        for i in idxs:
            answers.append(target[i])
            score_list.append(pred[i])
        # label 维度变为:[trans_dim]
        target = torch.stack(answers)
        # 进行维度重构 category维度:[category_nums, 1, trans_dim]
        pred = torch.stack(score_list)

        index = torch.zeros_like(pred, dtype=torch.uint8)  # 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # dim idx input
        index_float = index.type(torch.cuda.FloatTensor)  # 转 tensor
        ''' 以上的idx指示的应该是一个batch的y_true '''
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))

        x_m = pred - batch_m  # y 的 logit 减去 margin
        output = torch.where(index, x_m, pred)  # 按照修改位置合并
        return F.cross_entropy(self.s * output, target, weight=self.weight, ignore_index=-1)


def calculate_LMF_loss(focal_loss, ldam_loss, args):
    """
    计算LMF loss
    :param focal_loss: Focal Loss
    :param ldam_loss: LDAM Loss
    :param args:
    :return:
    """
    return args.lmf_hp1 * focal_loss + args.lmf_hp2 * ldam_loss
