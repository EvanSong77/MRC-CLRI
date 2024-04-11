# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:54
# @Author  : codewen77
import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from labels import get_aspect_category, get_sentiment
from losses.acos_losses import calculate_entity_loss, calculate_category_loss, calculate_sentiment_loss, \
    calculate_SCL_loss, FocalLoss
from metrics import ACOSScore
from question_template import get_English_Template
from tools import filter_unpaired, pair_combine, FGM, PGD, batch_pair_combine


class ACOSTrainer:
    def __init__(self, logger, model, optimizer, scheduler, tokenizer, args):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.args = args
        self.fgm = FGM(self.model)
        self.pgd = PGD(self.model)
        self.focalLoss = FocalLoss(self.args.flp_gamma)

    def train(self, train_dataloader, epoch):
        with tqdm(total=len(train_dataloader), desc="train") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                loss_sum = self.get_train_loss(batch)
                loss_sum.backward()

                # 使用FGM对抗训练
                if self.args.use_FGM:
                    # 在embedding层上添加对抗扰动
                    self.fgm.attack()
                    FGM_loss_sum = self.get_train_loss(batch)

                    # 恢复embedding参数
                    FGM_loss_sum.backward()
                    self.fgm.restore()

                # 使用PGD对抗训练
                if self.args.use_PGD:
                    self.pgd.backup_grad()
                    for t in range(self.args.pgd_k):
                        # 在embedding上添加对抗扰动, first attack时备份param.data
                        self.pgd.attack(is_first_attack=(t == 0))
                        if t != self.args.pgd_k - 1:
                            self.model.zero_grad()
                        else:
                            self.pgd.restore_grad()

                        PGD_loss_sum = self.get_train_loss(batch)
                        # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        PGD_loss_sum.backward()
                        # 恢复embedding参数
                    self.pgd.restore()

                # 梯度下降 更新参数
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                pbar.set_description(f'Epoch [{epoch}/{self.args.epoch_num}]')
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss_sum)})
                pbar.update(1)

    def eval(self, eval_dataloader):
        json_res = []
        acos_score = ACOSScore(self.logger)
        self.model.eval()
        Forward_Q1, Backward_Q1, Forward_Q2, Backward_Q2, Q3, Q4 = get_English_Template()
        f_asp_imp_start = 5
        b_opi_imp_start = 5
        for batch in tqdm(eval_dataloader):
            asp_predict, opi_predict, asp_opi_predict, triplets_predict, aocs_predict, quadruples_predict = [], [], [], [], [], []

            forward_pair_list, forward_pair_prob, forward_pair_ind_list = [], [], []

            backward_pair_list, backward_pair_prob, backward_pair_ind_list = [], [], []

            # forward q_1 nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
            passenge_index = batch.forward_asp_answer_start[0].gt(-1).float().nonzero()
            passenge = batch.forward_asp_query[0][passenge_index].squeeze(1)

            f_asp_start_scores, f_asp_end_scores = self.model(batch.forward_asp_query.cuda(),
                                                              batch.forward_asp_query_mask.cuda(),
                                                              batch.forward_asp_query_seg.cuda(), 0)
            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []
            for i in range(f_asp_start_ind.size(0)):
                if batch.forward_asp_answer_start[0, i] != -1:
                    if f_asp_start_ind[i].item() == 1:
                        f_asp_start_index_temp.append(i)
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                    if f_asp_end_ind[i].item() == 1:
                        f_asp_end_index_temp.append(i)
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

            f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
                f_asp_imp_start)

            for i in range(len(f_asp_start_index)):
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    opinion_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:7]])
                else:
                    opinion_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:6]])
                for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
                    opinion_query.append(batch.forward_asp_query[0][j].item())
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[7:]]))
                else:
                    opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[6:]]))
                imp_start = len(opinion_query)

                opinion_query_seg = [0] * len(opinion_query)
                f_opi_length = len(opinion_query)

                opinion_query = torch.tensor(opinion_query).long()
                opinion_query = torch.cat([opinion_query, passenge], -1).cuda().unsqueeze(0)
                opinion_query_seg += [1] * passenge.size(0)
                opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
                opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

                f_opi_start_scores, f_opi_end_scores = self.model(opinion_query, opinion_query_mask, opinion_query_seg,
                                                                  0)

                f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
                f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
                f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
                f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

                f_opi_start_prob_temp = []
                f_opi_end_prob_temp = []
                f_opi_start_index_temp = []
                f_opi_end_index_temp = []
                for k in range(f_opi_start_ind.size(0)):
                    if opinion_query_seg[0, k] == 1:
                        if f_opi_start_ind[k].item() == 1:
                            f_opi_start_index_temp.append(k)
                            f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                        if f_opi_end_ind[k].item() == 1:
                            f_opi_end_index_temp.append(k)
                            f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

                f_opi_start_index, f_opi_end_index, f_opi_prob = filter_unpaired(
                    f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, imp_start)

                for idx in range(len(f_opi_start_index)):
                    asp = [batch.forward_asp_query[0][j].item() for j in
                           range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                    opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                    # null -> -1, -1
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        asp_ind = [f_asp_start_index[i] - 8, f_asp_end_index[i] - 8]
                    else:
                        asp_ind = [f_asp_start_index[i] - 6, f_asp_end_index[i] - 6]
                    opi_ind = [f_opi_start_index[idx] - f_opi_length - 1, f_opi_end_index[idx] - f_opi_length - 1]
                    temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                    if asp_ind + opi_ind not in forward_pair_list:
                        forward_pair_list.append([asp] + [opi])
                        forward_pair_prob.append(temp_prob)
                        forward_pair_ind_list.append(asp_ind + opi_ind)

            # backward q_1
            b_opi_start_scores, b_opi_end_scores = self.model(batch.backward_opi_query.cuda(),
                                                              batch.backward_opi_query_mask.cuda(),
                                                              batch.backward_opi_query_seg.cuda(), 0)
            b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
            b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
            b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
            b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

            b_opi_start_prob_temp = []
            b_opi_end_prob_temp = []
            b_opi_start_index_temp = []
            b_opi_end_index_temp = []
            for i in range(b_opi_start_ind.size(0)):
                if batch.backward_opi_answer_start[0, i] != -1:
                    if b_opi_start_ind[i].item() == 1:
                        b_opi_start_index_temp.append(i)
                        b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                    if b_opi_end_ind[i].item() == 1:
                        b_opi_end_index_temp.append(i)
                        b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

            b_opi_start_index, b_opi_end_index, b_opi_prob = filter_unpaired(
                b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,
                b_opi_imp_start)

            # backward q_2
            for i in range(len(b_opi_start_index)):
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    aspect_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:7]])
                else:
                    aspect_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:6]])
                for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                    aspect_query.append(batch.backward_opi_query[0][j].item())

                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[7:]]))
                else:
                    aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[6:]]))
                imp_start = len(aspect_query)

                aspect_query_seg = [0] * len(aspect_query)
                b_asp_length = len(aspect_query)
                aspect_query = torch.tensor(aspect_query).long()
                aspect_query = torch.cat([aspect_query, passenge], -1).cuda().unsqueeze(0)
                aspect_query_seg += [1] * passenge.size(0)
                aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
                aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

                b_asp_start_scores, b_asp_end_scores = self.model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

                b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
                b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
                b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
                b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

                b_asp_start_prob_temp = []
                b_asp_end_prob_temp = []
                b_asp_start_index_temp = []
                b_asp_end_index_temp = []
                for k in range(b_asp_start_ind.size(0)):
                    if aspect_query_seg[0, k] == 1:
                        if b_asp_start_ind[k].item() == 1:
                            b_asp_start_index_temp.append(k)
                            b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                        if b_asp_end_ind[k].item() == 1:
                            b_asp_end_index_temp.append(k)
                            b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

                b_asp_start_index, b_asp_end_index, b_asp_prob = filter_unpaired(
                    b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, imp_start)

                for idx in range(len(b_asp_start_index)):
                    opi = [batch.backward_opi_query[0][j].item() for j in
                           range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                    asp = [aspect_query[0][j].item() for j in
                           range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                    # null -> -1, -1
                    asp_ind = [b_asp_start_index[idx] - b_asp_length - 1, b_asp_end_index[idx] - b_asp_length - 1]
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        opi_ind = [b_opi_start_index[i] - 8, b_opi_end_index[i] - 8]
                    else:
                        opi_ind = [b_opi_start_index[i] - 6, b_opi_end_index[i] - 6]
                    temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                    if asp_ind + opi_ind not in backward_pair_ind_list:
                        backward_pair_list.append([asp] + [opi])
                        backward_pair_prob.append(temp_prob)
                        backward_pair_ind_list.append(asp_ind + opi_ind)

            if self.args.use_Forward:
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
                for idx in range(len(forward_pair_list)):
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
            elif self.args.use_Backward:
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
                for idx in range(len(backward_pair_list)):
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
            else:
                # combine forward and backward pairs
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = pair_combine(forward_pair_list,
                                                                                                      forward_pair_prob,
                                                                                                      forward_pair_ind_list,
                                                                                                      backward_pair_list,
                                                                                                      backward_pair_prob,
                                                                                                      backward_pair_ind_list,
                                                                                                      self.args.alpha,
                                                                                                      self.args.beta)

            # category sentiment
            for idx in range(len(final_asp_list)):
                predict_opinion_num = len(final_opi_list[idx])
                # category sentiment
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    category_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:7]])
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:7]])
                else:
                    category_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:6]])
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]])
                category_query += final_asp_list[idx]
                sentiment_query += final_asp_list[idx]
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]]
                    category_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[7:8]])
                    sentiment_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[7:8]])
                else:
                    category_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[6:9]])
                    sentiment_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]])

                # 拼接opinion
                for idy in range(predict_opinion_num):
                    category_query2 = category_query + final_opi_list[idx][idy]
                    sentiment_query2 = sentiment_query + final_opi_list[idx][idy]
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        category_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[8:]]))
                        sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[8:]]))
                    else:
                        category_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[9:]]))
                        sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[9:]]))

                    category_query_seg = [0] * len(category_query2)
                    category_query2 = torch.tensor(category_query2).long().cuda()
                    category_query2 = torch.cat([category_query2, passenge.cuda()], -1).unsqueeze(0)
                    category_query_seg += [1] * passenge.size(0)
                    category_query_mask = torch.ones(category_query2.size(1)).float().cuda().unsqueeze(0)
                    category_query_seg = torch.tensor(category_query_seg).long().cuda().unsqueeze(0)

                    sentiment_query_seg = [0] * len(sentiment_query2)
                    sentiment_query2 = torch.tensor(sentiment_query2).long().cuda()
                    sentiment_query2 = torch.cat([sentiment_query2, passenge.cuda()], -1).unsqueeze(0)
                    sentiment_query_seg += [1] * passenge.size(0)
                    sentiment_query_mask = torch.ones(sentiment_query2.size(1)).float().cuda().unsqueeze(0)
                    sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

                    category_scores = self.model(category_query2, category_query_mask, category_query_seg, 1)
                    category_scores = F.softmax(category_scores, dim=1)
                    category_predicted = torch.argmax(category_scores[0], dim=0).item()

                    sentiment_scores = self.model(sentiment_query2, sentiment_query_mask, sentiment_query_seg, 2)
                    sentiment_scores = F.softmax(sentiment_scores, dim=1)
                    sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                    # 三元组、四元组组合
                    asp_f, opi_f = [], []
                    asp_f.append(final_asp_ind_list[idx][0])
                    asp_f.append(final_asp_ind_list[idx][1])
                    opi_f.append(final_opi_ind_list[idx][idy][0])
                    opi_f.append(final_opi_ind_list[idx][idy][1])
                    triplet_predict = [asp_f, opi_f, sentiment_predicted]
                    aoc_predict = [asp_f, opi_f, category_predicted]
                    quadruple_predict = [asp_f, category_predicted, opi_f, sentiment_predicted]

                    if asp_f not in asp_predict:
                        asp_predict.append(asp_f)
                    if opi_f not in opi_predict:
                        opi_predict.append(opi_f)
                    if [asp_f, opi_f] not in asp_opi_predict:
                        asp_opi_predict.append([asp_f, opi_f])
                    if triplet_predict not in triplets_predict:
                        triplets_predict.append(triplet_predict)
                    if aoc_predict not in aocs_predict:
                        aocs_predict.append(aoc_predict)
                    if quadruple_predict not in quadruples_predict:
                        quadruples_predict.append(quadruple_predict)

            acos_score.update(batch.aspects[0], batch.opinions[0], batch.pairs[0], batch.aste_triplets[0],
                              batch.aoc_triplets[0], batch.quadruples[0],
                              asp_predict, opi_predict, asp_opi_predict, triplets_predict, aocs_predict,
                              quadruples_predict)

            # sentences_list.append(' '.join(batch.sentence_token[0]))
            # pred_quads.append(quadruples_predict)
            # gold_quads.append(batch.quadruples[0])
            one_json = {'sentence': ' '.join(batch.sentence_token[0]), 'pred': str(quadruples_predict),
                        'gold': str(batch.quadruples[0])}
            json_res.append(one_json)
        with open(os.path.join(self.args.output_dir, self.args.task, self.args.data_type, 'pred.json'), 'w', encoding='utf-8') as fP:
            json.dump(json_res, fP, ensure_ascii=False, indent=4)
        return acos_score.compute()

    def batch_eval(self, eval_dataloader):
        start_time = time.time()
        json_res = []
        acos_score = ACOSScore(self.logger)
        self.model.eval()
        Forward_Q1, Backward_Q1, Forward_Q2, Backward_Q2, Q3, Q4 = get_English_Template()
        f_asp_imp_start = 5
        b_opi_imp_start = 5
        for batch in tqdm(eval_dataloader):
            forward_pair_list, forward_pair_prob, forward_pair_ind_list = [], [], []

            backward_pair_list, backward_pair_prob, backward_pair_ind_list = [], [], []

            # forward q_1 nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
            passenges = []
            for p in range(len(batch.forward_asp_answer_start)):
                passenge_index = batch.forward_asp_answer_start[p].gt(-1).float().nonzero()
                passenge = batch.forward_asp_query[p][passenge_index].squeeze(1)
                passenges.append(passenge)
            batch_size = len(passenges)
            # 进行第一轮
            forward_len = len(batch.forward_asp_query)
            turn1_query = torch.cat((batch.forward_asp_query, batch.backward_opi_query), dim=0)
            turn1_mask = torch.cat((batch.forward_asp_query_mask, batch.backward_opi_query_mask), dim=0)
            turn1_seg = torch.cat((batch.forward_asp_query_seg, batch.backward_opi_query_seg), dim=0)
            turn1_start_scores, turn1_end_scores = self.model(turn1_query.cuda(),
                                                              turn1_mask.cuda(),
                                                              turn1_seg.cuda(), 0)
            f_asp_start_scores, f_asp_end_scores = turn1_start_scores[:forward_len], turn1_end_scores[:forward_len]
            b_opi_start_scores, b_opi_end_scores = turn1_start_scores[forward_len:], turn1_end_scores[forward_len:]

            f_asp_start_scores = F.softmax(f_asp_start_scores, dim=-1)
            f_asp_end_scores = F.softmax(f_asp_end_scores, dim=-1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=-1)
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=-1)

            b_opi_start_scores = F.softmax(b_opi_start_scores, dim=-1)
            b_opi_end_scores = F.softmax(b_opi_end_scores, dim=-1)
            b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=-1)
            b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=-1)

            f_asp_start_indexs, f_asp_end_indexs, f_asp_probs = [], [], []
            for b in range(f_asp_end_prob.size(0)):
                f_asp_start_prob_temp = []
                f_asp_end_prob_temp = []
                f_asp_start_index_temp = []
                f_asp_end_index_temp = []
                for i in range(f_asp_start_ind[b].size(0)):
                    if batch.forward_asp_answer_start[b, i] != -1:
                        if f_asp_start_ind[b][i].item() == 1:
                            f_asp_start_index_temp.append(i)
                            f_asp_start_prob_temp.append(f_asp_start_prob[b][i].item())
                        if f_asp_end_ind[b][i].item() == 1:
                            f_asp_end_index_temp.append(i)
                            f_asp_end_prob_temp.append(f_asp_end_prob[b][i].item())

                f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
                    f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
                    f_asp_imp_start)
                f_asp_start_indexs.append(f_asp_start_index)
                f_asp_end_indexs.append(f_asp_end_index)
                f_asp_probs.append(f_asp_prob)

            # f_asp_start_indexs, f_asp_end_indexs, f_asp_probs
            f_asp_nums = []
            imp_starts = []
            f_opinion_querys, f_opinion_segs, f_opinion_masks = [], [], []
            for b in range(len(f_asp_start_indexs)):
                f_asp_nums.append(len(f_asp_start_indexs[b]))
                for i in range(len(f_asp_start_indexs[b])):
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        opinion_query = self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:7]])
                    else:
                        opinion_query = self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:6]])
                    for j in range(f_asp_start_indexs[b][i], f_asp_end_indexs[b][i] + 1):
                        opinion_query.append(batch.forward_asp_query[b][j].item())
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[7:]]))
                    else:
                        opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[6:]]))
                    imp_start = len(opinion_query)
                    imp_starts.append(imp_start)

                    opinion_query_seg = [0] * len(opinion_query)

                    opinion_query = torch.tensor(opinion_query).long()
                    opinion_query = torch.cat([opinion_query, passenges[b]], -1)
                    opinion_query_seg += [1] * passenges[b].size(0)
                    opinion_query_mask = torch.ones(opinion_query.size(0)).float()
                    opinion_query_seg = torch.tensor(opinion_query_seg).long()

                    f_opinion_querys.append(opinion_query)
                    f_opinion_segs.append(opinion_query_seg)
                    f_opinion_masks.append(opinion_query_mask)

            batch_f_ao_idxs = []
            if f_opinion_querys:
                # 进行padding
                f_opinion_querys = pad_sequence(f_opinion_querys, batch_first=True, padding_value=0).cuda()
                f_opinion_segs = pad_sequence(f_opinion_segs, batch_first=True, padding_value=1).cuda()
                f_opinion_masks = pad_sequence(f_opinion_masks, batch_first=True, padding_value=0).cuda()

                f_opi_start_scores, f_opi_end_scores = self.model(f_opinion_querys, f_opinion_masks, f_opinion_segs, 0)

                f_opi_start_scores = F.softmax(f_opi_start_scores, dim=-1)
                f_opi_end_scores = F.softmax(f_opi_end_scores, dim=-1)
                f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=-1)
                f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=-1)

                # 对asp进行batch处理
                b_f_asp_start_indexs = [asp for asps in f_asp_start_indexs for asp in asps]
                b_f_asp_end_indexs = [asp for asps in f_asp_end_indexs for asp in asps]
                b_f_asp_probs = [asp for asps in f_asp_probs for asp in asps]
                batch_map_list = [d for c in [[a] * f_asp_nums[a] for a in range(len(f_asp_nums))] for d in c]
                for b in range(f_opi_end_prob.size(0)):
                    f_opi_start_prob_temp = []
                    f_opi_end_prob_temp = []
                    f_opi_start_index_temp = []
                    f_opi_end_index_temp = []
                    for k in range(f_opi_start_ind[b].size(0)):
                        if f_opinion_segs[b, k].item() == 1:
                            if f_opi_start_ind[b][k].item() == 1:
                                f_opi_start_index_temp.append(k)
                                f_opi_start_prob_temp.append(f_opi_start_prob[b][k].item())
                            if f_opi_end_ind[b][k].item() == 1:
                                f_opi_end_index_temp.append(k)
                                f_opi_end_prob_temp.append(f_opi_end_prob[b][k].item())
                    f_opi_start_index, f_opi_end_index, f_opi_prob = filter_unpaired(
                        f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp,
                        imp_starts[b])
                    # 进行结果抽取
                    for idx in range(len(f_opi_start_index)):
                        asp = [batch.forward_asp_query[batch_map_list[b]][j].item() for j in
                               range(b_f_asp_start_indexs[b], b_f_asp_end_indexs[b] + 1)]
                        opi = [f_opinion_querys[b][j].item() for j in
                               range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                        # null -> -1, -1
                        if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                            asp_ind = [b_f_asp_start_indexs[b] - 8, b_f_asp_end_indexs[b] - 8]
                        else:
                            asp_ind = [b_f_asp_start_indexs[b] - 6, b_f_asp_end_indexs[b] - 6]
                        opi_ind = [f_opi_start_index[idx] - imp_starts[b] - 1, f_opi_end_index[idx] - imp_starts[b] - 1]
                        temp_prob = b_f_asp_probs[b] * f_opi_prob[idx]
                        if asp_ind + opi_ind not in forward_pair_list:
                            batch_f_ao_idxs.append(batch_map_list[b])
                            forward_pair_list.append([asp] + [opi])
                            forward_pair_prob.append(temp_prob)
                            forward_pair_ind_list.append(asp_ind + opi_ind)

            # backward q_1
            b_opi_start_indexs, b_opi_end_indexs, b_opi_probs = [], [], []
            for b in range(b_opi_end_prob.size(0)):
                b_opi_start_prob_temp = []
                b_opi_end_prob_temp = []
                b_opi_start_index_temp = []
                b_opi_end_index_temp = []
                for i in range(b_opi_start_ind[b].size(0)):
                    if batch.backward_opi_answer_start[b, i] != -1:
                        if b_opi_start_ind[b][i].item() == 1:
                            b_opi_start_index_temp.append(i)
                            b_opi_start_prob_temp.append(b_opi_start_prob[b][i].item())
                        if b_opi_end_ind[b][i].item() == 1:
                            b_opi_end_index_temp.append(i)
                            b_opi_end_prob_temp.append(b_opi_end_prob[b][i].item())

                b_opi_start_index, b_opi_end_index, b_opi_prob = filter_unpaired(
                    b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,
                    b_opi_imp_start)
                b_opi_start_indexs.append(b_opi_start_index)
                b_opi_end_indexs.append(b_opi_end_index)
                b_opi_probs.append(b_opi_prob)

            # backward q_2
            b_opi_nums = []
            imp_starts = []
            b_aspect_querys, b_aspect_segs, b_aspect_masks = [], [], []
            for b in range(len(b_opi_start_indexs)):
                b_opi_nums.append(len(b_opi_start_indexs[b]))
                for i in range(len(b_opi_start_indexs[b])):
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        aspect_query = self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:7]])
                    else:
                        aspect_query = self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:6]])
                    for j in range(b_opi_start_indexs[b][i], b_opi_end_indexs[b][i] + 1):
                        aspect_query.append(batch.backward_opi_query[b][j].item())
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[7:]]))
                    else:
                        aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[6:]]))
                    imp_start = len(aspect_query)
                    imp_starts.append(imp_start)

                    aspect_query_seg = [0] * len(aspect_query)

                    aspect_query = torch.tensor(aspect_query).long()
                    aspect_query = torch.cat([aspect_query, passenges[b]], -1)
                    aspect_query_seg += [1] * passenges[b].size(0)
                    aspect_query_mask = torch.ones(aspect_query.size(0)).float()
                    aspect_query_seg = torch.tensor(aspect_query_seg).long()

                    b_aspect_querys.append(aspect_query)
                    b_aspect_segs.append(aspect_query_seg)
                    b_aspect_masks.append(aspect_query_mask)

            batch_b_ao_idxs = []
            if b_aspect_querys:
                # 进行padding
                b_aspect_querys = pad_sequence(b_aspect_querys, batch_first=True, padding_value=0).cuda()
                b_aspect_segs = pad_sequence(b_aspect_segs, batch_first=True, padding_value=1).cuda()
                b_aspect_masks = pad_sequence(b_aspect_masks, batch_first=True, padding_value=0).cuda()

                b_asp_start_scores, b_asp_end_scores = self.model(b_aspect_querys, b_aspect_masks, b_aspect_segs, 0)

                b_asp_start_scores = F.softmax(b_asp_start_scores, dim=-1)
                b_asp_end_scores = F.softmax(b_asp_end_scores, dim=-1)
                b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=-1)
                b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=-1)

                # 对opi进行batch处理
                b_b_opi_start_indexs = [opi for opis in b_opi_start_indexs for opi in opis]
                b_b_opi_end_indexs = [opi for opis in b_opi_end_indexs for opi in opis]
                b_b_opi_probs = [opi for opis in b_opi_probs for opi in opis]
                f_batch_map_list = [d for c in [[a] * b_opi_nums[a] for a in range(len(b_opi_nums))] for d in c]
                for b in range(b_asp_end_prob.size(0)):
                    b_asp_start_prob_temp = []
                    b_asp_end_prob_temp = []
                    b_asp_start_index_temp = []
                    b_asp_end_index_temp = []
                    for k in range(b_asp_start_ind[b].size(0)):
                        if b_aspect_segs[b, k].item() == 1:
                            if b_asp_start_ind[b][k].item() == 1:
                                b_asp_start_index_temp.append(k)
                                b_asp_start_prob_temp.append(b_asp_start_prob[b][k].item())
                            if b_asp_end_ind[b][k].item() == 1:
                                b_asp_end_index_temp.append(k)
                                b_asp_end_prob_temp.append(b_asp_end_prob[b][k].item())

                    b_asp_start_index, b_asp_end_index, b_asp_prob = filter_unpaired(
                        b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp,
                        imp_starts[b])
                    # 进行结果抽取
                    for idx in range(len(b_asp_start_index)):
                        opi = [batch.backward_opi_query[f_batch_map_list[b]][j].item() for j in
                               range(b_b_opi_start_indexs[b], b_b_opi_end_indexs[b] + 1)]
                        asp = [b_aspect_querys[b][j].item() for j in
                               range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                        # null -> -1, -1
                        asp_ind = [b_asp_start_index[idx] - imp_starts[b] - 1, b_asp_end_index[idx] - imp_starts[b] - 1]
                        if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                            opi_ind = [b_b_opi_start_indexs[b] - 8, b_b_opi_end_indexs[b] - 8]
                        else:
                            opi_ind = [b_b_opi_start_indexs[b] - 6, b_b_opi_end_indexs[b] - 6]
                        temp_prob = b_asp_prob[idx] * b_b_opi_probs[b]
                        if asp_ind + opi_ind not in backward_pair_ind_list:
                            batch_b_ao_idxs.append(f_batch_map_list[b])
                            backward_pair_list.append([asp] + [opi])
                            backward_pair_prob.append(temp_prob)
                            backward_pair_ind_list.append(asp_ind + opi_ind)

            if self.args.use_Forward:
                batch_final_idxs = []
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
                for idx in range(len(forward_pair_list)):
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                        batch_final_idxs.append(batch_f_ao_idxs[idx])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
            elif self.args.use_Backward:
                batch_final_idxs = []
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
                for idx in range(len(backward_pair_list)):
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                        batch_final_idxs.append(batch_b_ao_idxs[idx])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
            else:
                # combine forward and backward pairs
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list, batch_final_idxs = batch_pair_combine(
                    forward_pair_list,
                    forward_pair_prob,
                    forward_pair_ind_list,
                    backward_pair_list,
                    backward_pair_prob,
                    backward_pair_ind_list,
                    batch_f_ao_idxs,
                    batch_b_ao_idxs,
                    batch_size,
                    self.args.alpha,
                    self.args.beta)

            # category sentiment
            ao_category_querys, ao_category_segs, ao_category_masks = [], [], []
            ao_sentiment_querys, ao_sentiment_segs, ao_sentiment_masks = [], [], []
            for idx in range(len(final_asp_list)):
                predict_opinion_num = len(final_opi_list[idx])
                # category sentiment
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    category_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:7]])
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:7]])
                else:
                    category_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:6]])
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]])
                category_query += final_asp_list[idx]
                sentiment_query += final_asp_list[idx]
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]]
                    category_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[7:8]])
                    sentiment_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[7:8]])
                else:
                    category_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[6:9]])
                    sentiment_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]])

                # 拼接opinion
                for idy in range(predict_opinion_num):
                    category_query2 = category_query + final_opi_list[idx][idy]
                    sentiment_query2 = sentiment_query + final_opi_list[idx][idy]
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        category_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[8:]]))
                        sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[8:]]))
                    else:
                        category_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[9:]]))
                        sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[9:]]))

                    category_query_seg = [0] * len(category_query2)
                    category_query2 = torch.tensor(category_query2).long()
                    category_query2 = torch.cat([category_query2, passenges[batch_final_idxs[idx]]], -1)
                    category_query_seg += [1] * passenges[batch_final_idxs[idx]].size(0)
                    category_query_mask = torch.ones(category_query2.size(0)).float()
                    category_query_seg = torch.tensor(category_query_seg).long()

                    sentiment_query_seg = [0] * len(sentiment_query2)
                    sentiment_query2 = torch.tensor(sentiment_query2).long()
                    sentiment_query2 = torch.cat([sentiment_query2, passenges[batch_final_idxs[idx]]], -1)
                    sentiment_query_seg += [1] * passenges[batch_final_idxs[idx]].size(0)
                    sentiment_query_mask = torch.ones(sentiment_query2.size(0)).float()
                    sentiment_query_seg = torch.tensor(sentiment_query_seg).long()

                    ao_category_querys.append(category_query2)
                    ao_category_segs.append(category_query_seg)
                    ao_category_masks.append(category_query_mask)

                    ao_sentiment_querys.append(sentiment_query2)
                    ao_sentiment_segs.append(sentiment_query_seg)
                    ao_sentiment_masks.append(sentiment_query_mask)

            if ao_category_querys:
                # 进行padding
                ao_category_querys = pad_sequence(ao_category_querys, batch_first=True, padding_value=0).cuda()
                ao_category_segs = pad_sequence(ao_category_segs, batch_first=True, padding_value=1).cuda()
                ao_category_masks = pad_sequence(ao_category_masks, batch_first=True, padding_value=0).cuda()

                ao_sentiment_querys = pad_sequence(ao_sentiment_querys, batch_first=True, padding_value=0).cuda()
                ao_sentiment_segs = pad_sequence(ao_sentiment_segs, batch_first=True, padding_value=1).cuda()
                ao_sentiment_masks = pad_sequence(ao_sentiment_masks, batch_first=True, padding_value=0).cuda()

                category_scores = self.model(ao_category_querys, ao_category_masks, ao_category_segs, 1)
                category_scores = F.softmax(category_scores, dim=-1)
                category_predicted = torch.argmax(category_scores, dim=-1)

                sentiment_scores = self.model(ao_sentiment_querys, ao_sentiment_masks, ao_sentiment_segs, 2)
                sentiment_scores = F.softmax(sentiment_scores, dim=-1)
                sentiment_predicted = torch.argmax(sentiment_scores, dim=-1)

                ao_nums = [len(fi) for fi in final_opi_list]
                ao_batch_map_list = [d for c in [[a] * ao_nums[a] for a in range(len(ao_nums))] for d in c]
                final_opi_ind_list = [opi for opis in final_opi_ind_list for opi in opis]
                # 三元组、四元组组合
                quadruples_predicts = []
                for idx in range(len(final_opi_ind_list)):
                    asp_f, opi_f = [], []
                    asp_f.append(final_asp_ind_list[ao_batch_map_list[idx]][0])
                    asp_f.append(final_asp_ind_list[ao_batch_map_list[idx]][1])
                    opi_f.append(final_opi_ind_list[idx][0])
                    opi_f.append(final_opi_ind_list[idx][1])
                    quadruple_predict = [asp_f, category_predicted[idx].item(), opi_f, sentiment_predicted[idx].item()]
                    if quadruple_predict not in quadruples_predicts:
                        quadruples_predicts.append(quadruple_predict)

                acos_score.update2(batch.quadruples, quadruples_predicts)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"*************************执行总耗时：{elapsed_time}秒*************************")
        return acos_score.compute2()

    def inference(self, reviews):
        self.model.eval()
        # category2id sentiment2id
        id2Category, id2Sentiment = get_aspect_category(self.args.task.lower(), self.args.data_type)[-1], \
            get_sentiment(self.args.task.lower())[-1]
        Forward_Q1, Backward_Q1, Forward_Q2, Backward_Q2, Q3, Q4 = get_English_Template()
        f_asp_imp_start = 5
        b_opi_imp_start = 5

        f_asp_query_list, f_asp_mask_list, f_asp_seg_list = [], [], []
        b_opi_query_list, b_opi_mask_list, b_opi_seg_list = [], [], []
        passenge_indexs, passenges = [], []
        f_max_len, b_max_len = 0, 0
        for review in reviews:
            review = review.split(' ')
            f_temp_text = Forward_Q1 + ["null"] + review
            f_temp_text = list(map(self.tokenizer.tokenize, f_temp_text))
            f_temp_text = [item for indices in f_temp_text for item in indices]
            passenge_indexs.append([i + len(Forward_Q1) for i in range(len(f_temp_text) - len(Forward_Q1))])
            passenges.append(self.tokenizer.convert_tokens_to_ids(["null"] + review))
            _forward_asp_query = self.tokenizer.convert_tokens_to_ids(f_temp_text)
            if f_max_len < len(_forward_asp_query):
                f_max_len = len(_forward_asp_query)
            f_asp_query_list.append(_forward_asp_query)
            _forward_asp_mask = [1 for _ in range(len(_forward_asp_query))]
            f_asp_mask_list.append(_forward_asp_mask)
            _forward_asp_seg = [0] * len(self.tokenizer.convert_tokens_to_ids(Forward_Q1)) + [1] * (
                    len(self.tokenizer.convert_tokens_to_ids(review)) + 1)
            f_asp_seg_list.append(_forward_asp_seg)

            b_temp_text = Backward_Q1 + ["null"] + review
            b_temp_text = list(map(self.tokenizer.tokenize, b_temp_text))
            b_temp_text = [item for indices in b_temp_text for item in indices]
            _backward_opi_query = self.tokenizer.convert_tokens_to_ids(b_temp_text)
            if b_max_len < len(_backward_opi_query):
                b_max_len = len(_backward_opi_query)
            b_opi_query_list.append(_backward_opi_query)
            _backward_opi_mask = [1 for _ in range(len(_backward_opi_query))]
            b_opi_mask_list.append(_backward_opi_mask)
            _backward_opi_seg = [0] * len(self.tokenizer.convert_tokens_to_ids(Backward_Q1)) + [1] * (
                    len(self.tokenizer.convert_tokens_to_ids(review)) + 1)
            b_opi_seg_list.append(_backward_opi_seg)
        # # 第一轮padding
        # for i in range(len(f_asp_query_list)):
        #     f_pad_len = f_max_len - len(f_asp_query_list[i])
        #     b_pad_len = b_max_len - len(b_opi_query_list[i])
        #
        #     f_asp_query_list[i].extend([0] * f_pad_len)
        #     b_opi_query_list[i].extend([0] * b_pad_len)
        #     f_asp_mask_list[i].extend([0] * f_pad_len)
        #     b_opi_mask_list[i].extend([0] * b_pad_len)
        #     f_asp_seg_list[i].extend([1] * f_pad_len)
        #     b_opi_seg_list[i].extend([1] * b_pad_len)

        for b in range(len(f_asp_query_list)):
            asp_predict, opi_predict, asp_opi_predict, triplets_predict, aocs_predict, quadruples_predict = [], [], [], [], [], []

            forward_pair_list, forward_pair_prob, forward_pair_ind_list = [], [], []

            backward_pair_list, backward_pair_prob, backward_pair_ind_list = [], [], []

            forward_asp_query = torch.tensor([f_asp_query_list[b]]).long()
            forward_asp_query_mask = torch.tensor([f_asp_mask_list[b]]).long()
            forward_asp_query_seg = torch.tensor([f_asp_seg_list[b]]).long()
            backward_opi_query = torch.tensor([b_opi_query_list[b]]).long()
            backward_opi_query_mask = torch.tensor([b_opi_mask_list[b]]).long()
            backward_opi_query_seg = torch.tensor([b_opi_seg_list[b]]).long()
            passenge = torch.tensor(passenges[b]).long()

            f_asp_start_scores, f_asp_end_scores = self.model(forward_asp_query.cuda(),
                                                              forward_asp_query_mask.cuda(),
                                                              forward_asp_query_seg.cuda(), 0)
            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)
            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []
            for i in range(f_asp_start_ind.size(0)):
                if i in passenge_indexs[b]:
                    if f_asp_start_ind[i].item() == 1:
                        f_asp_start_index_temp.append(i)
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                    if f_asp_end_ind[i].item() == 1:
                        f_asp_end_index_temp.append(i)
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

            f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
                f_asp_imp_start)

            for i in range(len(f_asp_start_index)):
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    opinion_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:7]])
                else:
                    opinion_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:6]])
                for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
                    opinion_query.append(forward_asp_query[0][j].item())
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[7:]]))
                else:
                    opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[6:]]))
                imp_start = len(opinion_query)

                opinion_query_seg = [0] * len(opinion_query)
                f_opi_length = len(opinion_query)

                opinion_query = torch.tensor(opinion_query).long()
                opinion_query = torch.cat([opinion_query, passenge], -1).cuda().unsqueeze(0)
                opinion_query_seg += [1] * passenge.size(0)
                opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
                opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

                f_opi_start_scores, f_opi_end_scores = self.model(opinion_query, opinion_query_mask, opinion_query_seg,
                                                                  0)

                f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
                f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
                f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
                f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

                f_opi_start_prob_temp = []
                f_opi_end_prob_temp = []
                f_opi_start_index_temp = []
                f_opi_end_index_temp = []
                for k in range(f_opi_start_ind.size(0)):
                    if opinion_query_seg[0, k] == 1:
                        if f_opi_start_ind[k].item() == 1:
                            f_opi_start_index_temp.append(k)
                            f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                        if f_opi_end_ind[k].item() == 1:
                            f_opi_end_index_temp.append(k)
                            f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

                f_opi_start_index, f_opi_end_index, f_opi_prob = filter_unpaired(
                    f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, imp_start)

                for idx in range(len(f_opi_start_index)):
                    asp = [forward_asp_query[0][j].item() for j in
                           range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                    opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                    # null -> -1, -1
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        asp_ind = [f_asp_start_index[i] - 8, f_asp_end_index[i] - 8]
                    else:
                        asp_ind = [f_asp_start_index[i] - 6, f_asp_end_index[i] - 6]
                    opi_ind = [f_opi_start_index[idx] - f_opi_length - 1, f_opi_end_index[idx] - f_opi_length - 1]
                    temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                    if asp_ind + opi_ind not in forward_pair_list:
                        forward_pair_list.append([asp] + [opi])
                        forward_pair_prob.append(temp_prob)
                        forward_pair_ind_list.append(asp_ind + opi_ind)

            # backward q_1
            b_opi_start_scores, b_opi_end_scores = self.model(backward_opi_query.cuda(),
                                                              backward_opi_query_mask.cuda(),
                                                              backward_opi_query_seg.cuda(), 0)
            b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
            b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
            b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
            b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

            b_opi_start_prob_temp = []
            b_opi_end_prob_temp = []
            b_opi_start_index_temp = []
            b_opi_end_index_temp = []
            for i in range(b_opi_start_ind.size(0)):
                if i in passenge_indexs[b]:
                    if b_opi_start_ind[i].item() == 1:
                        b_opi_start_index_temp.append(i)
                        b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                    if b_opi_end_ind[i].item() == 1:
                        b_opi_end_index_temp.append(i)
                        b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

            b_opi_start_index, b_opi_end_index, b_opi_prob = filter_unpaired(
                b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,
                b_opi_imp_start)

            # backward q_2
            for i in range(len(b_opi_start_index)):
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    aspect_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:7]])
                else:
                    aspect_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:6]])
                for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                    aspect_query.append(backward_opi_query[0][j].item())

                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[7:]]))
                else:
                    aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[6:]]))
                imp_start = len(aspect_query)

                aspect_query_seg = [0] * len(aspect_query)
                b_asp_length = len(aspect_query)
                aspect_query = torch.tensor(aspect_query).long()
                aspect_query = torch.cat([aspect_query, passenge], -1).cuda().unsqueeze(0)
                aspect_query_seg += [1] * passenge.size(0)
                aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
                aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

                b_asp_start_scores, b_asp_end_scores = self.model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

                b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
                b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
                b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
                b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

                b_asp_start_prob_temp = []
                b_asp_end_prob_temp = []
                b_asp_start_index_temp = []
                b_asp_end_index_temp = []
                for k in range(b_asp_start_ind.size(0)):
                    if aspect_query_seg[0, k] == 1:
                        if b_asp_start_ind[k].item() == 1:
                            b_asp_start_index_temp.append(k)
                            b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                        if b_asp_end_ind[k].item() == 1:
                            b_asp_end_index_temp.append(k)
                            b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

                b_asp_start_index, b_asp_end_index, b_asp_prob = filter_unpaired(
                    b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, imp_start)

                for idx in range(len(b_asp_start_index)):
                    opi = [backward_opi_query[0][j].item() for j in
                           range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                    asp = [aspect_query[0][j].item() for j in
                           range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                    # null -> -1, -1
                    asp_ind = [b_asp_start_index[idx] - b_asp_length - 1, b_asp_end_index[idx] - b_asp_length - 1]
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        opi_ind = [b_opi_start_index[i] - 8, b_opi_end_index[i] - 8]
                    else:
                        opi_ind = [b_opi_start_index[i] - 6, b_opi_end_index[i] - 6]
                    temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                    if asp_ind + opi_ind not in backward_pair_ind_list:
                        backward_pair_list.append([asp] + [opi])
                        backward_pair_prob.append(temp_prob)
                        backward_pair_ind_list.append(asp_ind + opi_ind)

            if self.args.use_Forward:
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
                for idx in range(len(forward_pair_list)):
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
            elif self.args.use_Backward:
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
                for idx in range(len(backward_pair_list)):
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
            else:
                # combine forward and backward pairs
                final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = pair_combine(forward_pair_list,
                                                                                                      forward_pair_prob,
                                                                                                      forward_pair_ind_list,
                                                                                                      backward_pair_list,
                                                                                                      backward_pair_prob,
                                                                                                      backward_pair_ind_list,
                                                                                                      self.args.alpha,
                                                                                                      self.args.beta)

            # category sentiment
            for idx in range(len(final_asp_list)):
                predict_opinion_num = len(final_opi_list[idx])
                # category sentiment
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    category_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:7]])
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:7]])
                else:
                    category_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:6]])
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]])
                category_query += final_asp_list[idx]
                sentiment_query += final_asp_list[idx]
                if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]]
                    category_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[7:8]])
                    sentiment_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[7:8]])
                else:
                    category_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[6:9]])
                    sentiment_query += self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]])

                # 拼接opinion
                for idy in range(predict_opinion_num):
                    category_query2 = category_query + final_opi_list[idx][idy]
                    sentiment_query2 = sentiment_query + final_opi_list[idx][idy]
                    if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
                        category_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[8:]]))
                        sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[8:]]))
                    else:
                        category_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[9:]]))
                        sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
                            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[9:]]))

                    category_query_seg = [0] * len(category_query2)
                    category_query2 = torch.tensor(category_query2).long().cuda()
                    category_query2 = torch.cat([category_query2, passenge.cuda()], -1).unsqueeze(0)
                    category_query_seg += [1] * passenge.size(0)
                    category_query_mask = torch.ones(category_query2.size(1)).float().cuda().unsqueeze(0)
                    category_query_seg = torch.tensor(category_query_seg).long().cuda().unsqueeze(0)

                    sentiment_query_seg = [0] * len(sentiment_query2)
                    sentiment_query2 = torch.tensor(sentiment_query2).long().cuda()
                    sentiment_query2 = torch.cat([sentiment_query2, passenge.cuda()], -1).unsqueeze(0)
                    sentiment_query_seg += [1] * passenge.size(0)
                    sentiment_query_mask = torch.ones(sentiment_query2.size(1)).float().cuda().unsqueeze(0)
                    sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

                    category_scores = self.model(category_query2, category_query_mask, category_query_seg, 1)
                    category_scores = F.softmax(category_scores, dim=1)
                    category_predicted = torch.argmax(category_scores[0], dim=0).item()

                    sentiment_scores = self.model(sentiment_query2, sentiment_query_mask, sentiment_query_seg, 2)
                    sentiment_scores = F.softmax(sentiment_scores, dim=1)
                    sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                    # 三元组、四元组组合
                    asp_f, opi_f = [], []
                    asp_f.append(final_asp_ind_list[idx][0])
                    asp_f.append(final_asp_ind_list[idx][1])
                    opi_f.append(final_opi_ind_list[idx][idy][0])
                    opi_f.append(final_opi_ind_list[idx][idy][1])
                    triplet_predict = [asp_f, opi_f, sentiment_predicted]
                    aoc_predict = [asp_f, opi_f, category_predicted]
                    quadruple_predict = [asp_f, category_predicted, opi_f, sentiment_predicted]

                    if asp_f not in asp_predict:
                        asp_predict.append(asp_f)
                    if opi_f not in opi_predict:
                        opi_predict.append(opi_f)
                    if [asp_f, opi_f] not in asp_opi_predict:
                        asp_opi_predict.append([asp_f, opi_f])
                    if triplet_predict not in triplets_predict:
                        triplets_predict.append(triplet_predict)
                    if aoc_predict not in aocs_predict:
                        aocs_predict.append(aoc_predict)
                    if quadruple_predict not in quadruples_predict:
                        quadruples_predict.append(quadruple_predict)

            print_quadruples_predict = []
            review_list = reviews[b].split(' ')
            tokenized_review = list(map(self.tokenizer.tokenize, review_list))
            subword_lengths = list(map(len, tokenized_review))
            token_start_idxs = np.cumsum([0] + subword_lengths[:-1])
            tokenized2word = {}
            for i in range(len(reviews[b].split(' '))):
                for j in range(token_start_idxs[i], token_start_idxs[i] + subword_lengths[i]):
                    tokenized2word[j] = i

            for q in quadruples_predict:
                if q[0] == [-1, -1]:
                    asp = 'NULL'
                else:
                    asp = ' '.join(review_list[tokenized2word[q[0][0]]:tokenized2word[q[0][-1]] + 1])
                if q[2] == [-1, -1]:
                    opi = 'NULL'
                else:
                    opi = ' '.join(review_list[tokenized2word[q[2][0]]:tokenized2word[q[2][-1]] + 1])
                cate, sent = id2Category[q[1]], id2Sentiment[q[-1]]
                print_quadruples_predict.append([asp, cate, opi, sent])
            print(f"`{reviews[b]}` 四元组抽取结果：`{print_quadruples_predict}`")

    def get_train_loss(self, batch):
        # 首先对数量进行处理 不需要为空的
        forward_opi_nums, backward_asp_nums, pairs_nums = batch.forward_opi_nums, batch.backward_asp_nums, batch.pairs_nums
        max_f_asp_len, max_f_opi_lens, max_b_opi_len, max_b_asp_lens, max_sent_cate_lens = max(
            batch.forward_aspect_len), \
            max([max(batch.forward_opinion_lens[b]) for b in range(self.args.train_batch_size)]), max(
            batch.backward_opinion_len), \
            max([max(batch.backward_aspects_lens[b]) for b in range(self.args.train_batch_size)]), \
            max([max(batch.sentiment_category_lens[b]) for b in range(self.args.train_batch_size)])

        forward_asp_query = batch.forward_asp_query[:, :max_f_asp_len]
        forward_asp_query_mask = batch.forward_asp_query_mask[:, :max_f_asp_len]
        forward_asp_query_seg = batch.forward_asp_query_seg[:, :max_f_asp_len]

        forward_asp_answer_start = batch.forward_asp_answer_start[:, :max_f_asp_len]
        forward_asp_answer_end = batch.forward_asp_answer_end[:, :max_f_asp_len]

        backward_opi_query = batch.backward_opi_query[:, :max_b_opi_len]
        backward_opi_query_mask = batch.backward_opi_query_mask[:, :max_b_opi_len]
        backward_opi_query_seg = batch.backward_opi_query_seg[:, :max_b_opi_len]

        backward_opi_answer_start = batch.backward_opi_answer_start[:, :max_b_opi_len]
        backward_opi_answer_end = batch.backward_opi_answer_end[:, :max_b_opi_len]

        forward_opi_query, forward_opi_query_mask, forward_opi_query_seg = [], [], []
        forward_opi_answer_start, forward_opi_answer_end = [], []
        backward_asp_query, backward_asp_query_mask, backward_asp_query_seg = [], [], []
        backward_asp_answer_start, backward_asp_answer_end = [], []
        category_query, category_query_mask, category_query_seg, category_answer = [], [], [], []
        sentiment_query, sentiment_query_mask, sentiment_query_seg, sentiment_answer = [], [], [], []
        for b in range(self.args.train_batch_size):
            # [nums, lens]
            forward_opi_query.append(batch.forward_opi_query[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_query_mask.append(batch.forward_opi_query_mask[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_query_seg.append(batch.forward_opi_query_seg[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_answer_start.append(batch.forward_opi_answer_start[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_answer_end.append(batch.forward_opi_answer_end[b][:forward_opi_nums[b], :max_f_opi_lens])

            backward_asp_query.append(batch.backward_asp_query[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_query_mask.append(batch.backward_asp_query_mask[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_query_seg.append(batch.backward_asp_query_seg[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_answer_start.append(batch.backward_asp_answer_start[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_answer_end.append(batch.backward_asp_answer_end[b][:backward_asp_nums[b], :max_b_asp_lens])

            category_query.append(batch.category_query[b][:pairs_nums[b], :max_sent_cate_lens])
            category_query_mask.append(batch.category_query_mask[b][:pairs_nums[b], :max_sent_cate_lens])
            category_query_seg.append(batch.category_query_seg[b][:pairs_nums[b], :max_sent_cate_lens])
            category_answer.append(batch.category_answer[b][:pairs_nums[b]])

            sentiment_query.append(batch.sentiment_query[b][:pairs_nums[b], :max_sent_cate_lens])
            sentiment_query_mask.append(batch.sentiment_query_mask[b][:pairs_nums[b], :max_sent_cate_lens])
            sentiment_query_seg.append(batch.sentiment_query_seg[b][:pairs_nums[b], :max_sent_cate_lens])
            sentiment_answer.append(batch.sentiment_answer[b][:pairs_nums[b]])

        forward_opi_query = torch.cat(forward_opi_query, dim=0)
        forward_opi_query_mask = torch.cat(forward_opi_query_mask, dim=0)
        forward_opi_query_seg = torch.cat(forward_opi_query_seg, dim=0)
        forward_opi_answer_start = torch.cat(forward_opi_answer_start, dim=0)
        forward_opi_answer_end = torch.cat(forward_opi_answer_end, dim=0)

        backward_asp_query = torch.cat(backward_asp_query, dim=0)
        backward_asp_query_mask = torch.cat(backward_asp_query_mask, dim=0)
        backward_asp_query_seg = torch.cat(backward_asp_query_seg, dim=0)
        backward_asp_answer_start = torch.cat(backward_asp_answer_start, dim=0)
        backward_asp_answer_end = torch.cat(backward_asp_answer_end, dim=0)

        category_query = torch.cat(category_query, dim=0)
        category_query_mask = torch.cat(category_query_mask, dim=0)
        category_query_seg = torch.cat(category_query_seg, dim=0)
        category_answer = torch.cat(category_answer, dim=0)

        sentiment_query = torch.cat(sentiment_query, dim=0)
        sentiment_query_mask = torch.cat(sentiment_query_mask, dim=0)
        sentiment_query_seg = torch.cat(sentiment_query_seg, dim=0)
        sentiment_answer = torch.cat(sentiment_answer, dim=0)

        if self.args.use_Forward:
            # q1_f
            f_aspect_start_scores, f_aspect_end_scores = self.model(forward_asp_query.cuda(),
                                                                    forward_asp_query_mask.cuda(),
                                                                    forward_asp_query_seg.cuda(), 0)
            # q2_f
            f_opi_start_scores, f_opi_end_scores = self.model(forward_opi_query.cuda(), forward_opi_query_mask.cuda(),
                                                              forward_opi_query_seg.cuda(), 0)
            # loss 计算
            f_asp_loss = calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                                               forward_asp_answer_start.cuda(), forward_asp_answer_end.cuda())
            f_opi_loss = calculate_entity_loss(f_opi_start_scores, f_opi_end_scores, forward_opi_answer_start.cuda(),
                                               forward_opi_answer_end.cuda())
        elif self.args.use_Backward:
            # q1_b
            b_opi_start_scores, b_opi_end_scores = self.model(backward_opi_query.cuda(), backward_opi_query_mask.cuda(),
                                                              backward_opi_query_seg.cuda(), 0)
            # q2_b
            b_asp_start_scores, b_asp_end_scores = self.model(backward_asp_query.cuda(), backward_asp_query_mask.cuda(),
                                                              backward_asp_query_seg.cuda(), 0)
            b_opi_loss = calculate_entity_loss(b_opi_start_scores, b_opi_end_scores, backward_opi_answer_start.cuda(),
                                               backward_opi_answer_end.cuda())
            b_asp_loss = calculate_entity_loss(b_asp_start_scores, b_asp_end_scores, backward_asp_answer_start.cuda(),
                                               backward_asp_answer_end.cuda())
        else:
            # q1_f
            f_aspect_start_scores, f_aspect_end_scores = self.model(forward_asp_query.cuda(),
                                                                    forward_asp_query_mask.cuda(),
                                                                    forward_asp_query_seg.cuda(), 0)
            # q2_f
            f_opi_start_scores, f_opi_end_scores = self.model(forward_opi_query.cuda(), forward_opi_query_mask.cuda(),
                                                              forward_opi_query_seg.cuda(), 0)
            # q1_b
            b_opi_start_scores, b_opi_end_scores = self.model(backward_opi_query.cuda(), backward_opi_query_mask.cuda(),
                                                              backward_opi_query_seg.cuda(), 0)
            # q2_b
            b_asp_start_scores, b_asp_end_scores = self.model(backward_asp_query.cuda(), backward_asp_query_mask.cuda(),
                                                              backward_asp_query_seg.cuda(), 0)
            # loss 计算
            f_asp_loss = calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                                               forward_asp_answer_start.cuda(), forward_asp_answer_end.cuda())
            f_opi_loss = calculate_entity_loss(f_opi_start_scores, f_opi_end_scores, forward_opi_answer_start.cuda(),
                                               forward_opi_answer_end.cuda())

            b_opi_loss = calculate_entity_loss(b_opi_start_scores, b_opi_end_scores, backward_opi_answer_start.cuda(),
                                               backward_opi_answer_end.cuda())
            b_asp_loss = calculate_entity_loss(b_asp_start_scores, b_asp_end_scores, backward_asp_answer_start.cuda(),
                                               backward_asp_answer_end.cuda())

        # q_3
        category_scores = self.model(category_query.cuda(), category_query_mask.cuda(), category_query_seg.cuda(), 1)
        # q_4
        sentiment_scores = self.model(sentiment_query.cuda(), sentiment_query_mask.cuda(), sentiment_query_seg.cuda(),
                                      2)

        if self.args.use_FocalLoss:
            category_loss = self.focalLoss(category_scores, category_answer.cuda())
            sentiment_loss = self.focalLoss(sentiment_scores, sentiment_answer.cuda())
        else:
            # 交叉熵loss
            category_loss = calculate_category_loss(category_scores, category_answer.cuda())
            sentiment_loss = calculate_sentiment_loss(sentiment_scores, sentiment_answer.cuda())

        # 使用对比loss
        if self.args.use_category_SCL:
            scl_category_loss = calculate_SCL_loss(category_answer.cuda(), category_scores)
            all_category_loss = (
                                        1 - self.args.contrastive_lr1) * category_loss + self.args.contrastive_lr1 * scl_category_loss
        else:
            all_category_loss = category_loss
        if self.args.use_sentiment_SCL:
            scl_sentiment_loss = calculate_SCL_loss(sentiment_answer.cuda(), sentiment_scores)
            all_sentiment_loss = (
                                         1 - self.args.contrastive_lr2) * sentiment_loss + self.args.contrastive_lr2 * scl_sentiment_loss
        else:
            all_sentiment_loss = sentiment_loss

        # 正常训练loss
        if self.args.use_Forward:
            loss_sum = (f_asp_loss + f_opi_loss) + 2 * all_category_loss + 3 * all_sentiment_loss
        elif self.args.use_Backward:
            loss_sum = (b_opi_loss + b_asp_loss) + 2 * all_category_loss + 3 * all_sentiment_loss
        else:
            loss_sum = (f_asp_loss + f_opi_loss) + (
                    b_opi_loss + b_asp_loss) + 2 * all_category_loss + 3 * all_sentiment_loss
        return loss_sum
