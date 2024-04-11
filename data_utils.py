# -*- coding: utf-8 -*-
# @Time    : 2022/11/6 14:55
# @Author  : codewen77
import random

import numpy as np
from torch.utils.data import Dataset

from labels import get_aspect_category, get_sentiment
from question_template import get_English_Template
from samples import DataSample, TokenizedSample


def getJsonl(data_path):
    """
    从jsonl文件获取数据
    :param data_path: train dev test jsonl数据存放路径
    :return:
    """
    with open(data_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    data_list = []
    # count = 0
    for line in lines:
        line = eval(line)
        # if len(line['sentence']) > 200:
        #     count += 1
        #     continue
        data_list.append(line)
    return data_list


def get_quadruples(lines, tokenizer, task):
    # Line sample:
    # {'sentence': 'get the tuna of gari .', 'labels': [[(2, 5), 'FOOD#QUALITY', (-1, -1), 'positive']]}
    # {'sentence': '东西挺好用的，保湿度强', 'labels': [[(-1, -1), '整体', (2, 6), '正面'], [(7, 10), '功效', (10, 11), '正面']]}
    sentence_token = []
    quadruple_list = []
    for line in lines:
        sentence, labels = line['sentence'], line['labels']

        # 全部转为小写
        if task.lower() == "asqe" or task.lower() == 'zh_quad':
            word_list = list(sentence.lower())
        else:
            word_list = sentence.lower().split()
        # 使用分词器进行处理
        subwords_token = list(map(tokenizer.tokenize, word_list))
        subword_lengths = list(map(len, subwords_token))
        subwords_token = [item for indices in subwords_token for item in indices]
        token_start_idxs = np.cumsum([0] + subword_lengths[:-1])

        quad = []
        for label in labels:
            if label[0] == (-1, -1):
                asp = (-1, -1)
            else:
                asp_start, asp_end = token_start_idxs[label[0][0]], token_start_idxs[label[0][1] - 1] + subword_lengths[
                    label[0][1] - 1] - 1
                asp = (asp_start, asp_end)
            if label[2] == (-1, -1):
                opi = (-1, -1)
            else:
                opi_start, opi_end = token_start_idxs[label[2][0]], token_start_idxs[label[2][1] - 1] + subword_lengths[
                    label[2][1] - 1] - 1
                opi = (opi_start, opi_end)
            category, sentiment = label[1], label[-1]

            quad.append((asp, category, opi, sentiment))

        sentence_token.append(subwords_token)
        quadruple_list.append(quad)
    return sentence_token, quadruple_list


def deal_quadruple(quadruple, category_dict, sentiment_dict):
    aspects = []
    opinions = []
    pairs = []
    aste_triplets = []
    aoc_triplets = []
    quadruples = []

    f_quadruple_aspect = []
    f_quadruple_opinion = []
    b_quadruple_aspect = []
    b_quadruple_opinion = []
    quadruple_category = []
    quadruple_sentiment = []
    for t in quadruple:
        if t[0] not in f_quadruple_aspect:
            f_quadruple_aspect.append(t[0])
            f_quadruple_opinion.append([t[2]])
            quadruple_category.append([category_dict[t[1]]])
            quadruple_sentiment.append([sentiment_dict[t[-1]]])
        else:
            idx = f_quadruple_aspect.index(t[0])
            f_quadruple_opinion[idx].append(t[2])
            quadruple_category[idx].append(category_dict[t[1]])
            quadruple_sentiment[idx].append(sentiment_dict[t[-1]])
        if t[2] not in b_quadruple_opinion:
            b_quadruple_opinion.append(t[2])
            b_quadruple_aspect.append([t[0]])
        else:
            idx = b_quadruple_opinion.index(t[2])
            b_quadruple_aspect[idx].append(t[0])

        asp = list(t[0])
        opi = list(t[2])
        pai = [asp, opi]
        trip = [asp, opi, sentiment_dict[t[-1]]]
        aoc = [asp, opi, category_dict[t[1]]]
        quad = [asp, category_dict[t[1]], opi, sentiment_dict[t[-1]]]
        if asp not in aspects:
            aspects.append(asp)
        if opi not in opinions:
            opinions.append(opi)
        if pai not in pairs:
            pairs.append(pai)
        if trip not in aste_triplets:
            aste_triplets.append(trip)
        if aoc not in aoc_triplets:
            aoc_triplets.append(aoc)
        if quad not in quadruples:
            quadruples.append(quad)

    return f_quadruple_aspect, f_quadruple_opinion, b_quadruple_aspect, b_quadruple_opinion, quadruple_category, quadruple_sentiment, \
        aspects, opinions, pairs, aste_triplets, aoc_triplets, quadruples


class ACOSDataset(Dataset):
    def __init__(self, tokenizer, args, dataset_type):
        """
        :param tokenizer: 分词器
        :param data_path: 数据存放路径
        :param dataset_type: 数据集类型(train、dev、test)
        :param task: 任务
        :param data_type:数据类型
        """
        # 分词器
        self.tokenizer = tokenizer
        data_path = args.data_path
        task = args.task
        data_type = args.data_type

        self.max_fopi_nums, self.max_basp_nums, self.max_pair_nums = 0, 0, 0
        self.max_fasp_len, self.max_fopi_len, self.max_basp_len, self.max_bopi_len, self.max_pair_len = 0, 0, 0, 0, 0
        low_resource = args.low_resource

        self.data_samples = self._build_examples(data_path, dataset_type, task, data_type)
        datas_len = len(self.data_samples)
        self.datas_len = int(low_resource * datas_len)
        if dataset_type == 'train' and low_resource != 1.0:
            # 低资源环境
            sample_indices = random.sample(list(range(0, datas_len)), self.datas_len)
            temps = self._build_tokenized()
            self.tokenized_samples = [temps[i] for i in sample_indices]
        else:
            self.tokenized_samples = self._build_tokenized()

    def __getitem__(self, item):
        return self.tokenized_samples[item]

    def __len__(self):
        return len(self.tokenized_samples)

    def _build_examples(self, data_path, dataset_type, task, data_type):
        """
        :param data_path: 数据存放路径
        :param dataset_type: 数据集类型(train、dev、test)
        :param task: 任务(acos、quad)
        :param data_type:数据类型([rest, laptop]、[rest15, rest16])
        :return:
        """
        data_samples = []

        # category2id sentiment2id
        category2id, sentiment2id = get_aspect_category(task, data_type)[1], get_sentiment(task)[1]
        # lines data
        lines = getJsonl(data_path + dataset_type + '.jsonl')
        # get quadruples
        sentence_token, quadruple_list = get_quadruples(lines, self.tokenizer, task)

        # 英文模板
        Forward_Q1, Backward_Q1, Forward_Q2, Backward_Q2, Q3, Q4 = get_English_Template()
        # ================================question and answer================================
        for k in range(len(sentence_token)):
            quadruple = quadruple_list[k]
            text = sentence_token[k]

            f_quadruple_aspect, f_quadruple_opinion, b_quadruple_aspect, b_quadruple_opinion, quadruple_category, quadruple_sentiment, \
                aspects, opinions, pairs, aste_triplets, aoc_triplets, quadruples = deal_quadruple(quadruple,
                                                                                                   category2id,
                                                                                                   sentiment2id)
            forward_query_list = []
            forward_answer_list = []
            backward_query_list = []
            backward_answer_list = []

            category_query_list = []
            category_answer_list = []
            sentiment_query_list = []
            sentiment_answer_list = []

            # forward
            # aspect query
            forward_query_list.append(Forward_Q1)
            if len(forward_query_list[0]) + 1 + len(text) > self.max_fasp_len:
                self.max_fasp_len = len(forward_query_list[0]) + 1 + len(text)
            start = [0] * (len(text) + 1)
            end = [0] * (len(text) + 1)
            for ta in f_quadruple_aspect:
                if ta == (-1, -1):
                    start[0] = 1
                    end[0] = 1
                else:
                    start[ta[0] + 1] = 1
                    end[ta[-1] + 1] = 1
            forward_answer_list.append([start, end])

            # opinion query
            for idx in range(len(f_quadruple_aspect)):
                ta = f_quadruple_aspect[idx]
                if ta == (-1, -1):
                    if task.lower() == "asqe" or task.lower() == 'zh_quad':
                        query = Forward_Q2[0:7] + ["null"] + Forward_Q2[7:]
                    else:
                        query = Forward_Q2[0:6] + ["null"] + Forward_Q2[6:]
                else:
                    if task.lower() == "asqe" or task.lower() == 'zh_quad':
                        query = Forward_Q2[0:7] + text[ta[0]:ta[-1] + 1] + Forward_Q2[7:]
                    else:
                        query = Forward_Q2[0:6] + text[ta[0]:ta[-1] + 1] + Forward_Q2[6:]
                forward_query_list.append(query)
                if len(query) + 1 + len(text) > self.max_fopi_len:
                    self.max_fopi_len = len(query) + 1 + len(text)
                start = [0] * (len(text) + 1)
                end = [0] * (len(text) + 1)
                for to in f_quadruple_opinion[idx]:
                    if to == (-1, -1):
                        start[0] = 1
                        end[0] = 1
                    else:
                        start[to[0] + 1] = 1
                        end[to[-1] + 1] = 1
                forward_answer_list.append([start, end])

                # category query
                # sentiment query
                if ta == (-1, -1):
                    if task.lower() == "asqe" or task.lower() == 'zh_quad':
                        query1 = Q3[0:7] + ["null"] + Q3[7:8]
                        query2 = Q4[0:7] + ["null"] + Q4[7:8]
                    else:
                        query1 = Q3[0:6] + ["null"] + Q3[6:9]
                        query2 = Q4[0:6] + ["null"] + Q4[6:9]
                else:
                    if task.lower() == "asqe" or task.lower() == 'zh_quad':
                        query1 = Q3[0:7] + text[ta[0]:ta[-1] + 1] + Q3[7:8]
                        query2 = Q4[0:7] + text[ta[0]:ta[-1] + 1] + Q4[7:8]
                    else:
                        query1 = Q3[0:6] + text[ta[0]:ta[-1] + 1] + Q3[6:9]
                        query2 = Q4[0:6] + text[ta[0]:ta[-1] + 1] + Q4[6:9]
                for idy in range(len(f_quadruple_opinion[idx])):
                    to = f_quadruple_opinion[idx][idy]
                    if to == (-1, -1):
                        if task.lower() == "asqe" or task.lower() == 'zh_quad':
                            query1 = query1 + ["null"] + Q3[8:]
                            query2 = query2 + ["null"] + Q4[8:]
                        else:
                            query1 = query1 + ["null"] + Q3[9:]
                            query2 = query2 + ["null"] + Q4[9:]
                    else:
                        if task.lower() == "asqe" or task.lower() == 'zh_quad':
                            query1 = query1 + text[to[0]:to[-1] + 1] + Q3[8:]
                            query2 = query2 + text[to[0]:to[-1] + 1] + Q4[8:]
                        else:
                            query1 = query1 + text[to[0]:to[-1] + 1] + Q3[9:]
                            query2 = query2 + text[to[0]:to[-1] + 1] + Q4[9:]
                    if len(query1) + 1 + len(text) > self.max_pair_len:
                        self.max_pair_len = len(query1) + 1 + len(text)
                    category_query_list.append(query1)
                    category_answer_list.append(quadruple_category[idx][idy])
                    sentiment_query_list.append(query2)
                    sentiment_answer_list.append(quadruple_sentiment[idx][idy])

            # backward
            # opinion query
            backward_query_list.append(Backward_Q1)
            if len(backward_query_list[0]) + 1 + len(text) > self.max_bopi_len:
                self.max_bopi_len = len(backward_query_list[0]) + 1 + len(text)
            start = [0] * (len(text) + 1)
            end = [0] * (len(text) + 1)
            for to in b_quadruple_opinion:
                if to == (-1, -1):
                    start[0] = 1
                    end[0] = 1
                else:
                    start[to[0] + 1] = 1
                    end[to[-1] + 1] = 1
            backward_answer_list.append([start, end])
            # aspect query
            for idx in range(len(b_quadruple_opinion)):
                ta = b_quadruple_opinion[idx]
                if ta == (-1, -1):
                    if task.lower() == "asqe" or task.lower() == 'zh_quad':
                        query = Backward_Q2[0:7] + ["null"] + Backward_Q2[7:]
                    else:
                        query = Backward_Q2[0:6] + ["null"] + Backward_Q2[6:]
                else:
                    if task.lower() == "asqe" or task.lower() == 'zh_quad':
                        query = Backward_Q2[0:7] + text[ta[0]:ta[-1] + 1] + Backward_Q2[7:]
                    else:
                        query = Backward_Q2[0:6] + text[ta[0]:ta[-1] + 1] + Backward_Q2[6:]
                backward_query_list.append(query)
                if len(query) + 1 + len(text) > self.max_basp_len:
                    self.max_basp_len = len(query) + 1 + len(text)
                start = [0] * (len(text) + 1)
                end = [0] * (len(text) + 1)
                for to in b_quadruple_aspect[idx]:
                    if to == (-1, -1):
                        start[0] = 1
                        end[0] = 1
                    else:
                        start[to[0] + 1] = 1
                        end[to[-1] + 1] = 1
                backward_answer_list.append([start, end])

            # forward (max_opinion_nums)
            if len(forward_query_list) - 1 > self.max_fopi_nums:
                self.max_fopi_nums = len(forward_query_list) - 1
            # backward (max_aspect_nums)
            if len(backward_query_list) - 1 > self.max_basp_nums:
                self.max_basp_nums = len(backward_query_list) - 1
            # max_pair_nums
            if len(category_query_list) > self.max_pair_nums:
                self.max_pair_nums = len(category_query_list)

            sample = DataSample(text, aspects, opinions, pairs, aste_triplets, aoc_triplets, quadruples,
                                forward_query_list, forward_answer_list, backward_query_list, backward_answer_list,
                                category_query_list, category_answer_list, sentiment_query_list, sentiment_answer_list)
            data_samples.append(sample)

        return data_samples

    def _build_tokenized(self):
        tokenized_samples = []
        for item in range(len(self.data_samples)):
            # ======================进行token化处理======================
            _forward_asp_query, _forward_asp_answer_start, _forward_asp_answer_end, _forward_asp_query_mask, _forward_asp_query_seg = [], [], [], [], []
            _forward_opi_query, _forward_opi_answer_start, _forward_opi_answer_end, _forward_opi_query_mask, _forward_opi_query_seg = [], [], [], [], []

            _backward_asp_query, _backward_asp_answer_start, _backward_asp_answer_end, _backward_asp_query_mask, _backward_asp_query_seg = [], [], [], [], []
            _backward_opi_query, _backward_opi_answer_start, _backward_opi_answer_end, _backward_opi_query_mask, _backward_opi_query_seg = [], [], [], [], []

            _category_query, _category_answer, _category_query_mask, _category_query_seg = [], [], [], []
            _sentiment_query, _sentiment_answer, _sentiment_query_mask, _sentiment_query_seg = [], [], [], []

            sample = self.data_samples[item]
            sentence_token = sample.sentence_token
            forward_querys, forward_answers = sample.forward_querys, sample.forward_answers
            backward_querys, backward_answers = sample.backward_querys, sample.backward_answers

            category_querys = sample.category_querys
            category_answers = sample.category_answers
            sentiment_querys = sample.sentiment_querys
            sentiment_answers = sample.sentiment_answers

            # forward opi query nums
            forward_pad_num = len(forward_querys) - 1
            # backward asp query nums
            backward_pad_num = len(backward_querys) - 1

            # Forward
            # aspect query
            temp_text = forward_querys[0] + ["null"] + sentence_token
            f_asp_pad_len = self.max_fasp_len - len(temp_text)
            forward_aspect_len = len(temp_text)

            temp_answer_start = [-1] * len(forward_querys[0]) + forward_answers[0][0]
            temp_answer_end = [-1] * len(forward_querys[0]) + forward_answers[0][1]
            temp_query_seg = [0] * len(forward_querys[0]) + [1] * (len(sentence_token) + 1)
            assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

            # padding
            # query
            _forward_asp_query = self.tokenizer.convert_tokens_to_ids(temp_text)
            _forward_asp_query.extend([0] * f_asp_pad_len)
            # query_mask
            _forward_asp_query_mask = [1 for i in range(len(temp_text))]
            _forward_asp_query_mask.extend([0] * f_asp_pad_len)
            # seg
            _forward_asp_query_seg = temp_query_seg
            _forward_asp_query_seg.extend([1] * f_asp_pad_len)
            # answer
            _forward_asp_answer_start = temp_answer_start
            _forward_asp_answer_start.extend([-1] * f_asp_pad_len)
            _forward_asp_answer_end = temp_answer_end
            _forward_asp_answer_end.extend([-1] * f_asp_pad_len)

            # opinion query
            forward_opinion_lens = []
            for i in range(1, len(forward_querys)):
                temp_text = forward_querys[i] + ["null"] + sentence_token
                pad_len = self.max_fopi_len - len(temp_text)
                forward_opinion_lens.append(len(temp_text))

                temp_answer_start = [-1] * len(forward_querys[i]) + forward_answers[i][0]
                temp_answer_end = [-1] * len(forward_querys[i]) + forward_answers[i][1]
                temp_query_seg = [0] * len(forward_querys[i]) + [1] * (len(sentence_token) + 1)
                assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

                # padding
                # query
                single_opinion_query = self.tokenizer.convert_tokens_to_ids(temp_text)
                single_opinion_query.extend([0] * pad_len)
                # query_mask
                single_opinion_query_mask = [1 for i in range(len(temp_text))]
                single_opinion_query_mask.extend([0] * pad_len)
                # query_seg
                single_opinion_query_seg = temp_query_seg
                single_opinion_query_seg.extend([1] * pad_len)
                # answer
                single_opinion_answer_start = temp_answer_start
                single_opinion_answer_start.extend([-1] * pad_len)
                single_opinion_answer_end = temp_answer_end
                single_opinion_answer_end.extend([-1] * pad_len)

                _forward_opi_query.append(single_opinion_query)
                _forward_opi_query_mask.append(single_opinion_query_mask)
                _forward_opi_query_seg.append(single_opinion_query_seg)
                _forward_opi_answer_start.append(single_opinion_answer_start)
                _forward_opi_answer_end.append(single_opinion_answer_end)

            # PAD: max_opi_num
            _forward_opi_query.extend([[0 for i in range(self.max_fopi_len)]] * (self.max_fopi_nums - forward_pad_num))
            _forward_opi_query_mask.extend(
                [[0 for i in range(self.max_fopi_len)]] * (self.max_fopi_nums - forward_pad_num))
            _forward_opi_query_seg.extend(
                [[0 for i in range(self.max_fopi_len)]] * (self.max_fopi_nums - forward_pad_num))
            _forward_opi_answer_start.extend(
                [[-1 for i in range(self.max_fopi_len)]] * (self.max_fopi_nums - forward_pad_num))
            _forward_opi_answer_end.extend(
                [[-1 for i in range(self.max_fopi_len)]] * (self.max_fopi_nums - forward_pad_num))
            assert len(_forward_opi_query) == len(_forward_opi_query_mask) == len(_forward_opi_query_seg) == len(
                _forward_opi_answer_start) == len(_forward_opi_answer_end) == self.max_fopi_nums

            # Backward
            # opinion
            # query
            temp_text = backward_querys[0] + ["null"] + sentence_token
            b_opi_pad_len = self.max_bopi_len - len(temp_text)
            backward_opinion_len = len(temp_text)

            temp_answer_start = [-1] * len(backward_querys[0]) + backward_answers[0][0]
            temp_answer_end = [-1] * len(backward_querys[0]) + backward_answers[0][1]
            temp_query_seg = [0] * len(backward_querys[0]) + [1] * (len(sentence_token) + 1)
            assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

            # padding
            # query
            _backward_opi_query = self.tokenizer.convert_tokens_to_ids(temp_text)
            _backward_opi_query.extend([0] * b_opi_pad_len)
            # mask
            _backward_opi_query_mask = [1 for i in range(len(temp_text))]
            _backward_opi_query_mask.extend([0] * b_opi_pad_len)
            # seg
            _backward_opi_query_seg = temp_query_seg
            _backward_opi_query_seg.extend([1] * b_opi_pad_len)
            # answer
            _backward_opi_answer_start = temp_answer_start
            _backward_opi_answer_start.extend([-1] * b_opi_pad_len)
            _backward_opi_answer_end = temp_answer_end
            _backward_opi_answer_end.extend([-1] * b_opi_pad_len)

            # Aspect
            backward_aspects_lens = []
            for i in range(1, len(backward_querys)):
                temp_text = backward_querys[i] + ["null"] + sentence_token
                pad_len = self.max_basp_len - len(temp_text)
                backward_aspects_lens.append(len(temp_text))

                temp_answer_start = [-1] * len(backward_querys[i]) + backward_answers[i][0]
                temp_answer_end = [-1] * len(backward_querys[i]) + backward_answers[i][1]
                temp_query_seg = [0] * len(backward_querys[i]) + [1] * (len(sentence_token) + 1)
                assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

                # padding
                # query
                single_aspect_query = self.tokenizer.convert_tokens_to_ids(temp_text)
                single_aspect_query.extend([0] * pad_len)
                # query_mask
                single_aspect_query_mask = [1 for i in range(len(temp_text))]
                single_aspect_query_mask.extend([0] * pad_len)
                # query_seg
                single_aspect_query_seg = temp_query_seg
                single_aspect_query_seg.extend([1] * pad_len)
                # answer
                single_aspect_answer_start = temp_answer_start
                single_aspect_answer_start.extend([-1] * pad_len)
                single_aspect_answer_end = temp_answer_end
                single_aspect_answer_end.extend([-1] * pad_len)

                _backward_asp_query.append(single_aspect_query)
                _backward_asp_query_mask.append(single_aspect_query_mask)
                _backward_asp_query_seg.append(single_aspect_query_seg)
                _backward_asp_answer_start.append(single_aspect_answer_start)
                _backward_asp_answer_end.append(single_aspect_answer_end)

            # PAD: max_aspect_num
            _backward_asp_query.extend(
                [[0 for i in range(self.max_basp_len)]] * (self.max_basp_nums - backward_pad_num))
            _backward_asp_query_mask.extend(
                [[0 for i in range(self.max_basp_len)]] * (self.max_basp_nums - backward_pad_num))
            _backward_asp_query_seg.extend(
                [[0 for i in range(self.max_basp_len)]] * (self.max_basp_nums - backward_pad_num))
            _backward_asp_answer_start.extend(
                [[-1 for i in range(self.max_basp_len)]] * (self.max_basp_nums - backward_pad_num))
            _backward_asp_answer_end.extend(
                [[-1 for i in range(self.max_basp_len)]] * (self.max_basp_nums - backward_pad_num))
            assert len(_backward_asp_query) == len(_backward_asp_query_mask) == len(_backward_asp_query_seg) == len(
                _backward_asp_answer_start) == len(_backward_asp_answer_end) == self.max_basp_nums

            # category
            sentiment_category_lens = []
            assert len(category_querys) == len(sentiment_querys)
            for i in range(len(category_querys)):
                question_tokenized = category_querys[i] + ["null"] + sentence_token
                question_tokenized2 = sentiment_querys[i] + ["null"] + sentence_token
                pad_len = self.max_pair_len - len(question_tokenized)
                pad_len2 = self.max_pair_len - len(question_tokenized2)
                assert len(question_tokenized) == len(question_tokenized2)
                sentiment_category_lens.append(len(question_tokenized))

                # mask
                question_mask = [1] * len(question_tokenized)
                question_mask2 = [1] * len(question_tokenized2)
                # segment
                question_seg = [0] * len(category_querys[i]) + [1] * (len(sentence_token) + 1)
                question_seg2 = [0] * len(sentiment_querys[i]) + [1] * (len(sentence_token) + 1)
                # answer
                answer = category_answers[i]
                answer2 = sentiment_answers[i]

                # padding
                # query
                question_tokenized = self.tokenizer.convert_tokens_to_ids(question_tokenized)
                question_tokenized.extend([0] * pad_len)
                question_tokenized2 = self.tokenizer.convert_tokens_to_ids(question_tokenized2)
                question_tokenized2.extend([0] * pad_len2)
                # query mask
                question_mask.extend([0 for i in range(pad_len)])
                question_mask2.extend([0 for i in range(pad_len2)])
                # query seg
                question_seg.extend([1] * pad_len)
                question_seg2.extend([1] * pad_len2)

                assert len(question_tokenized) == len(question_mask) == len(question_seg)
                assert len(question_tokenized2) == len(question_mask2) == len(question_seg2)

                _category_query_mask.append(question_mask)
                _category_query_seg.append(question_seg)
                _category_answer.append(answer)
                _category_query.append(question_tokenized)
                _sentiment_query_mask.append(question_mask2)
                _sentiment_query_seg.append(question_seg2)
                _sentiment_answer.append(answer2)
                _sentiment_query.append(question_tokenized2)

            # PAD: max_pair_nums
            _category_query.extend(
                [[0 for i in range(self.max_pair_len)]] * (self.max_pair_nums - len(category_querys)))
            _category_query_mask.extend(
                [[0 for i in range(self.max_pair_len)]] * (self.max_pair_nums - len(category_querys)))
            _category_query_seg.extend(
                [[0 for i in range(self.max_pair_len)]] * (self.max_pair_nums - len(category_querys)))
            _category_answer.extend([-1] * (self.max_pair_nums - len(category_querys)))
            assert len(_category_query) == len(_category_query_mask) == len(_category_query_seg) == len(
                _category_answer) == self.max_pair_nums
            _sentiment_query.extend(
                [[0 for i in range(self.max_pair_len)]] * (self.max_pair_nums - len(sentiment_querys)))
            _sentiment_query_mask.extend(
                [[0 for i in range(self.max_pair_len)]] * (self.max_pair_nums - len(sentiment_querys)))
            _sentiment_query_seg.extend(
                [[0 for i in range(self.max_pair_len)]] * (self.max_pair_nums - len(sentiment_querys)))
            _sentiment_answer.extend([-1] * (self.max_pair_nums - len(sentiment_querys)))
            assert len(_sentiment_query) == len(_sentiment_query_mask) == len(_sentiment_query_seg) == len(
                _sentiment_answer) == self.max_pair_nums

            assert len(category_querys) == len(sentiment_querys)
            sample = TokenizedSample(sentence_token, len(sentence_token),
                                     sample.aspects, sample.opinions, sample.pairs, sample.aste_triplets,
                                     sample.aoc_triplets, sample.quadruples,
                                     _forward_asp_query, _forward_asp_answer_start, _forward_asp_answer_end,
                                     _forward_asp_query_mask, _forward_asp_query_seg,
                                     _forward_opi_query, _forward_opi_answer_start, _forward_opi_answer_end,
                                     _forward_opi_query_mask, _forward_opi_query_seg,
                                     _backward_asp_query, _backward_asp_answer_start, _backward_asp_answer_end,
                                     _backward_asp_query_mask, _backward_asp_query_seg,
                                     _backward_opi_query, _backward_opi_answer_start, _backward_opi_answer_end,
                                     _backward_opi_query_mask, _backward_opi_query_seg,
                                     _category_query, _category_answer, _category_query_mask, _category_query_seg,
                                     _sentiment_query, _sentiment_answer, _sentiment_query_mask, _sentiment_query_seg,
                                     forward_pad_num, backward_pad_num, len(category_querys),
                                     forward_aspect_len, forward_opinion_lens, backward_opinion_len,
                                     backward_aspects_lens, sentiment_category_lens)
            tokenized_samples.append(sample)

        return tokenized_samples