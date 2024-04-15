# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:57
# @Author  : codewen77
import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def seed_everything(seed=1024):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def filter_unpaired(start_prob, end_prob, start, end, imp_start):
    filtered_start = []
    filtered_end = []
    filtered_prob = []
    if len(start) > 0 and len(end) > 0:
        length = start[-1] + 1 if start[-1] >= end[-1] else end[-1] + 1
        temp_seq = [0] * length
        for s in start:
            temp_seq[s] += 1
        for e in end:
            temp_seq[e] += 2
        last_start = -1
        for idx in range(len(temp_seq)):
            assert temp_seq[idx] < 4
            # 确定start
            if temp_seq[idx] == 1:
                last_start = idx
            # start和end不重合
            elif temp_seq[idx] == 2:
                if last_start != -1 and idx - last_start <= 5:
                    if last_start == imp_start and idx != last_start:
                        continue
                    filtered_start.append(last_start)
                    filtered_end.append(idx)
                    prob = start_prob[start.index(last_start)] * end_prob[end.index(idx)]
                    filtered_prob.append(prob)
                last_start = -1
            # start和end重合
            elif temp_seq[idx] == 3:
                if last_start == imp_start and idx != last_start:
                    continue
                filtered_start.append(idx)
                filtered_end.append(idx)
                prob = start_prob[start.index(idx)] * end_prob[end.index(idx)]
                filtered_prob.append(prob)
                last_start = -1
    return filtered_start, filtered_end, filtered_prob


def pair_combine(forward_pair_list, forward_pair_prob, forward_pair_ind_list,
                 backward_pair_list, backward_pair_prob, backward_pair_ind_list,
                 alpha, beta=0):
    forward_pair_list2, forward_pair_prob2, forward_pair_ind_list2 = [], [], []
    backward_pair_list2, backward_pair_prob2, backward_pair_ind_list2 = [], [], []
    # 先进行距离修剪
    if beta == 0:
        forward_pair_list2, forward_pair_prob2, forward_pair_ind_list2 = forward_pair_list, forward_pair_prob, forward_pair_ind_list
        backward_pair_list2, backward_pair_prob2, backward_pair_ind_list2 = backward_pair_list, backward_pair_prob, backward_pair_ind_list
    else:
        assert len(forward_pair_list) == len(forward_pair_prob) == len(forward_pair_ind_list)
        assert len(backward_pair_list) == len(backward_pair_prob) == len(backward_pair_ind_list)
        for idx in range(len(forward_pair_ind_list)):
            if min_distance(forward_pair_ind_list[idx][0], forward_pair_ind_list[idx][1],
                            forward_pair_ind_list[idx][2], forward_pair_ind_list[idx][-1]) <= beta:
                forward_pair_list2.append(forward_pair_list[idx])
                forward_pair_prob2.append(forward_pair_prob[idx])
                forward_pair_ind_list2.append(forward_pair_ind_list[idx])
        for idx in range(len(backward_pair_ind_list)):
            if min_distance(backward_pair_ind_list[idx][0], backward_pair_ind_list[idx][1],
                            backward_pair_ind_list[idx][2], backward_pair_ind_list[idx][-1]) <= beta:
                backward_pair_list2.append(backward_pair_list[idx])
                backward_pair_prob2.append(backward_pair_prob[idx])
                backward_pair_ind_list2.append(backward_pair_ind_list[idx])

    # forward
    final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
    for idx in range(len(forward_pair_list2)):
        if forward_pair_list2[idx] in backward_pair_list2:
            if forward_pair_list2[idx][0] not in final_asp_list:
                final_asp_list.append(forward_pair_list2[idx][0])
                final_opi_list.append([forward_pair_list2[idx][1]])
                final_asp_ind_list.append(forward_pair_ind_list2[idx][:2])
                final_opi_ind_list.append([forward_pair_ind_list2[idx][2:]])
            else:
                asp_index = final_asp_list.index(forward_pair_list2[idx][0])
                if forward_pair_list2[idx][1] not in final_opi_list[asp_index]:
                    final_opi_list[asp_index].append(forward_pair_list2[idx][1])
                    final_opi_ind_list[asp_index].append(forward_pair_ind_list2[idx][2:])
        else:
            if forward_pair_prob2[idx] >= alpha:
                if forward_pair_list2[idx][0] not in final_asp_list:
                    final_asp_list.append(forward_pair_list2[idx][0])
                    final_opi_list.append([forward_pair_list2[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list2[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list2[idx][2:]])
                else:
                    asp_index = final_asp_list.index(forward_pair_list2[idx][0])
                    if forward_pair_list2[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(forward_pair_list2[idx][1])
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list2[idx][2:])
    # backward
    for idx in range(len(backward_pair_list2)):
        if backward_pair_list2[idx] not in forward_pair_list2:
            if backward_pair_prob2[idx] >= alpha:
                if backward_pair_list2[idx][0] not in final_asp_list:
                    final_asp_list.append(backward_pair_list2[idx][0])
                    final_opi_list.append([backward_pair_list2[idx][1]])
                    final_asp_ind_list.append(backward_pair_ind_list2[idx][:2])
                    final_opi_ind_list.append([backward_pair_ind_list2[idx][2:]])
                else:
                    asp_index = final_asp_list.index(backward_pair_list2[idx][0])
                    if backward_pair_list2[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(backward_pair_list2[idx][1])
                        final_opi_ind_list[asp_index].append(backward_pair_ind_list2[idx][2:])

    return final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list


def batch_deal(forward_pair_list2, forward_pair_prob2, forward_pair_ind_list2, batch_f_ao_idxs,
               backward_pair_list2, backward_pair_prob2, backward_pair_ind_list2, batch_b_ao_idxs, batch_size):
    forward_pair_list3 = defaultdict(list)
    forward_pair_prob3 = defaultdict(list)
    forward_pair_ind_list3 = defaultdict(list)
    backward_pair_list3 = defaultdict(list)
    backward_pair_prob3 = defaultdict(list)
    backward_pair_ind_list3 = defaultdict(list)
    # 对forward进行batch处理
    for i in range(len(forward_pair_list2)):
        forward_pair_list3[batch_f_ao_idxs[i]].append(forward_pair_list2[i])
        forward_pair_prob3[batch_f_ao_idxs[i]].append(forward_pair_prob2[i])
        forward_pair_ind_list3[batch_f_ao_idxs[i]].append(forward_pair_ind_list2[i])
    # 对backward进行batch处理
    for i in range(len(backward_pair_list2)):
        backward_pair_list3[batch_b_ao_idxs[i]].append(backward_pair_list2[i])
        backward_pair_prob3[batch_b_ao_idxs[i]].append(backward_pair_prob2[i])
        backward_pair_ind_list3[batch_b_ao_idxs[i]].append(backward_pair_ind_list2[i])

    for i in range(batch_size):
        if i not in batch_f_ao_idxs:
            forward_pair_list3[i] = []
            forward_pair_prob3[i] = []
            forward_pair_ind_list3[i] = []
        if i not in batch_b_ao_idxs:
            backward_pair_list3[i] = []
            backward_pair_prob3[i] = []
            backward_pair_ind_list3[i] = []
    forward_pair_list3 = {k: forward_pair_list3[k] for k in sorted(forward_pair_list3)}
    forward_pair_prob3 = {k: forward_pair_prob3[k] for k in sorted(forward_pair_prob3)}
    forward_pair_ind_list3 = {k: forward_pair_ind_list3[k] for k in sorted(forward_pair_ind_list3)}
    backward_pair_list3 = {k: backward_pair_list3[k] for k in sorted(backward_pair_list3)}
    backward_pair_prob3 = {k: backward_pair_prob3[k] for k in sorted(backward_pair_prob3)}
    backward_pair_ind_list3 = {k: backward_pair_ind_list3[k] for k in sorted(backward_pair_ind_list3)}
    forward_pair_list4 = [v for v in forward_pair_list3.values()]
    forward_pair_prob4 = [v for v in forward_pair_prob3.values()]
    forward_pair_ind_list4 = [v for v in forward_pair_ind_list3.values()]
    backward_pair_list4 = [v for v in backward_pair_list3.values()]
    backward_pair_prob4 = [v for v in backward_pair_prob3.values()]
    backward_pair_ind_list4 = [v for v in backward_pair_ind_list3.values()]

    return forward_pair_list4, forward_pair_prob4, forward_pair_ind_list4, backward_pair_list4, backward_pair_prob4, \
        backward_pair_ind_list4


def batch_pair_combine(forward_pair_list, forward_pair_prob, forward_pair_ind_list,
                       backward_pair_list, backward_pair_prob, backward_pair_ind_list,
                       batch_f_ao_idxs, batch_b_ao_idxs, batch_size, alpha, beta=0):
    # 进行beta过滤
    if beta == 0:
        forward_pair_list2, forward_pair_prob2, forward_pair_ind_list2 = forward_pair_list, forward_pair_prob, forward_pair_ind_list
        backward_pair_list2, backward_pair_prob2, backward_pair_ind_list2 = backward_pair_list, backward_pair_prob, backward_pair_ind_list
        temp_batch_f_ao_idxs, temp_batch_b_ao_idxs = batch_f_ao_idxs, batch_b_ao_idxs
    else:
        forward_pair_list2, forward_pair_prob2, forward_pair_ind_list2 = [], [], []
        backward_pair_list2, backward_pair_prob2, backward_pair_ind_list2 = [], [], []
        temp_batch_f_ao_idxs, temp_batch_b_ao_idxs = [], []
        assert len(forward_pair_list) == len(forward_pair_prob) == len(forward_pair_ind_list)
        assert len(backward_pair_list) == len(backward_pair_prob) == len(backward_pair_ind_list)
        for idx in range(len(forward_pair_ind_list)):
            if min_distance(forward_pair_ind_list[idx][0], forward_pair_ind_list[idx][1],
                            forward_pair_ind_list[idx][2], forward_pair_ind_list[idx][-1]) <= beta:
                forward_pair_list2.append(forward_pair_list[idx])
                forward_pair_prob2.append(forward_pair_prob[idx])
                forward_pair_ind_list2.append(forward_pair_ind_list[idx])
                temp_batch_f_ao_idxs.append(batch_f_ao_idxs[idx])
        for idx in range(len(backward_pair_ind_list)):
            if min_distance(backward_pair_ind_list[idx][0], backward_pair_ind_list[idx][1],
                            backward_pair_ind_list[idx][2], backward_pair_ind_list[idx][-1]) <= beta:
                backward_pair_list2.append(backward_pair_list[idx])
                backward_pair_prob2.append(backward_pair_prob[idx])
                backward_pair_ind_list2.append(backward_pair_ind_list[idx])
                temp_batch_b_ao_idxs.append(batch_b_ao_idxs[idx])

    forward_pair_list4, forward_pair_prob4, forward_pair_ind_list4, backward_pair_list4, backward_pair_prob4, backward_pair_ind_list4 = batch_deal(forward_pair_list2, forward_pair_prob2, forward_pair_ind_list2, temp_batch_f_ao_idxs, backward_pair_list2, backward_pair_prob2, backward_pair_ind_list2, temp_batch_b_ao_idxs, batch_size)
    # 进行alpha过滤
    final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
    batch_final_idxs = []
    # forward
    for b in range(len(forward_pair_list4)):
        batch_final_idx = []
        final_asps, final_opis, final_asp_inds, final_opi_inds = [], [], [], []
        for idx in range(len(forward_pair_list4[b])):
            if forward_pair_list4[b][idx] in backward_pair_list4[b]:
                if forward_pair_list4[b][idx][0] not in final_asps:
                    final_asps.append(forward_pair_list4[b][idx][0])
                    final_opis.append([forward_pair_list4[b][idx][1]])
                    final_asp_inds.append(forward_pair_ind_list4[b][idx][:2])
                    final_opi_inds.append([forward_pair_ind_list4[b][idx][2:]])
                    batch_final_idx.append(b)
                else:
                    asp_index = final_asps.index(forward_pair_list4[b][idx][0])
                    if forward_pair_list4[b][idx][1] not in final_opis[asp_index]:
                        final_opis[asp_index].append(forward_pair_list4[b][idx][1])
                        final_opi_inds[asp_index].append(forward_pair_ind_list4[b][idx][2:])
            else:
                if forward_pair_prob4[b][idx] >= alpha:
                    if forward_pair_list4[b][idx][0] not in final_asps:
                        final_asps.append(forward_pair_list4[b][idx][0])
                        final_opis.append([forward_pair_list4[b][idx][1]])
                        final_asp_inds.append(forward_pair_ind_list4[b][idx][:2])
                        final_opi_inds.append([forward_pair_ind_list4[b][idx][2:]])
                        batch_final_idx.append(b)
                    else:
                        asp_index = final_asps.index(forward_pair_list4[b][idx][0])
                        if forward_pair_list4[b][idx][1] not in final_opis[asp_index]:
                            final_opis[asp_index].append(forward_pair_list4[b][idx][1])
                            final_opi_inds[asp_index].append(forward_pair_ind_list4[b][idx][2:])
        batch_final_idxs.append(batch_final_idx)
        final_asp_list.append(final_asps)
        final_opi_list.append(final_opis)
        final_asp_ind_list.append(final_asp_inds)
        final_opi_ind_list.append(final_opi_inds)
    # backward
    for b in range(len(backward_pair_list4)):
        batch_final_idx = []
        final_asps, final_opis, final_asp_inds, final_opi_inds = [], [], [], []
        for idx in range(len(backward_pair_list4[b])):
            if backward_pair_list4[b][idx] not in forward_pair_list4[b]:
                if backward_pair_prob4[b][idx] >= alpha:
                    if backward_pair_list4[b][idx][0] not in final_asps:
                        final_asps.append(backward_pair_list4[b][idx][0])
                        final_opis.append([backward_pair_list4[b][idx][1]])
                        final_asp_inds.append(backward_pair_ind_list4[b][idx][:2])
                        final_opi_inds.append([backward_pair_ind_list4[b][idx][2:]])
                        batch_final_idx.append(b)
                    else:
                        asp_index = final_asps.index(backward_pair_list4[b][idx][0])
                        if backward_pair_list4[b][idx][1] not in final_opis[asp_index]:
                            final_opis[asp_index].append(backward_pair_list4[b][idx][1])
                            final_opi_inds[asp_index].append(backward_pair_ind_list4[b][idx][2:])
        batch_final_idxs[b] += batch_final_idx
        final_asp_list[b] += final_asps
        final_opi_list[b] += final_opis
        final_asp_ind_list[b] += final_asp_inds
        final_opi_ind_list[b] += final_opi_inds
    # 转化回去
    batch_final_idxs = [i for i, b in enumerate(batch_final_idxs) for _ in b]
    final_asp_list = [i for b in final_asp_list for i in b]
    final_opi_list = [i for b in final_opi_list for i in b]
    final_asp_ind_list = [i for b in final_asp_ind_list for i in b]
    final_opi_ind_list = [i for b in final_opi_ind_list for i in b]
    return final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list, batch_final_idxs


def min_distance(a, b, c, d):
    return min(abs(b - c), abs(a - d))


def save_model(save_path, save_type, epoch, optimizer, model):
    # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    # save_path = os.path.join(save_path, "{}_model.pth".format(save_type))
    # torch.save(state, save_path)
    # 修改为pkl格式
    save_path = os.path.join(save_path, "{}_model.pkl".format(save_type))
    torch.save(model, save_path)
    return save_path


def print_results(logger, results):
    """
    results {
                "aspect": {"precision": asp_p, "recall": asp_r, "f1": asp_f},
                "opinion": {"precision": opi_p, "recall": opi_r, "f1": opi_f},
                "ao_pair": {"precision": ao_pair_p, "recall": ao_pair_r, "f1": ao_pair_f},
                "aste_triplet": {"precision": aste_triplet_p, "recall": aste_triplet_r, "f1": aste_triplet_f},
                "aoc_triplet": {"precision": aoc_triplet_p, "recall": aoc_triplet_r, "f1": aoc_triplet_f},
                "imp_quadruple": {"precision": imp_quadruple_p, "recall": imp_quadruple_r, "f1": imp_quadruple_f},
                "quadruple": {"precision": quadruple_p, "recall": quadruple_r, "f1": quadruple_f}
            }
    """
    logger.info("aspect: {}".format(results['aspect']))
    logger.info("opinion: {}".format(results['opinion']))
    logger.info("ao_pair: {}".format(results['ao_pair']))
    logger.info("aste_triplet: {}".format(results['aste_triplet']))
    logger.info("aoc_triplet: {}".format(results['aoc_triplet']))
    logger.info("imp_quadruple: {}".format(results['imp_quadruple']))
    logger.info("quadruple: {}".format(results['quadruple']))


def print_results2(logger, results):
    """
    results {
                "aspect": {"precision": asp_p, "recall": asp_r, "f1": asp_f},
                "opinion": {"precision": opi_p, "recall": opi_r, "f1": opi_f},
                "ao_pair": {"precision": ao_pair_p, "recall": ao_pair_r, "f1": ao_pair_f},
                "aste_triplet": {"precision": aste_triplet_p, "recall": aste_triplet_r, "f1": aste_triplet_f},
                "aoc_triplet": {"precision": aoc_triplet_p, "recall": aoc_triplet_r, "f1": aoc_triplet_f},
                "imp_quadruple": {"precision": imp_quadruple_p, "recall": imp_quadruple_r, "f1": imp_quadruple_f},
                "quadruple": {"precision": quadruple_p, "recall": quadruple_r, "f1": quadruple_f}
            }
    """
    # logger.info("aste_triplet: {}".format(results['aste_triplet']))
    # logger.info("aoc_triplet: {}".format(results['aoc_triplet']))
    logger.info("quadruple: {}".format(results['quadruple']))


class FGM():
    """
        FGM对抗训练 Example：
        # 初始化
        fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
        for batch_input, batch_label in data:
            # 正常训练
            loss = model(batch_input, batch_label)
            loss.backward() # 反向传播，得到正常的grad
            # 对抗训练
            fgm.attack() # 在embedding上添加对抗扰动
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
        """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model, emb_name='emb.', epsilon=1., alpha=0.3):
        """
        PGD对抗训练 Example：
        pgd = PGD(model)
        K = 3 # 一般设置为3
        for batch_input, batch_label in data:
            # 正常训练
            loss = model(batch_input, batch_label)
            loss.backward() # 反向传播，得到正常的grad
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = model(batch_input, batch_label)
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
        """
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
