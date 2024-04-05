# -*- coding: utf-8 -*-
# @Time    : 2022/11/6 16:22
# @Author  : codewen77


class DataSample(object):
    def __init__(self,
                 sentence_token,
                 aspects, opinions, pairs, aste_triplets, aoc_triplets, quadruples,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 category_querys,
                 category_answers,
                 sentiment_querys,
                 sentiment_answers):
        self.sentence_token = sentence_token
        self.aspects = aspects
        self.opinions = opinions
        self.pairs = pairs
        self.aste_triplets = aste_triplets
        self.aoc_triplets = aoc_triplets
        self.quadruples = quadruples

        self.forward_querys = forward_querys
        self.forward_answers = forward_answers
        self.backward_querys = backward_querys
        self.backward_answers = backward_answers

        self.category_querys = category_querys
        self.category_answers = category_answers
        self.sentiment_querys = sentiment_querys
        self.sentiment_answers = sentiment_answers


class TokenizedSample(object):
    def __init__(self,
                 sentence_token, sentence_len,
                 aspects, opinions, pairs, aste_triplets, aoc_triplets, quadruples,
                 _forward_asp_query, _forward_asp_answer_start, _forward_asp_answer_end, _forward_asp_query_mask, _forward_asp_query_seg,
                 _forward_opi_query, _forward_opi_answer_start, _forward_opi_answer_end, _forward_opi_query_mask, _forward_opi_query_seg,
                 _backward_asp_query, _backward_asp_answer_start, _backward_asp_answer_end, _backward_asp_query_mask, _backward_asp_query_seg,
                 _backward_opi_query, _backward_opi_answer_start, _backward_opi_answer_end, _backward_opi_query_mask, _backward_opi_query_seg,
                 _category_query, _category_answer, _category_query_mask, _category_query_seg,
                 _sentiment_query, _sentiment_answer, _sentiment_query_mask, _sentiment_query_seg,
                 _forward_opi_nums, _backward_asp_nums, _pairs_nums,
                 forward_aspect_len, forward_opinion_lens, backward_opinion_len, backward_aspects_lens, sentiment_category_lens):
        self.sentence_token = sentence_token
        self.sentence_len = sentence_len

        self.aspects = aspects
        self.opinions = opinions
        self.pairs = pairs
        self.aste_triplets = aste_triplets
        self.aoc_triplets = aoc_triplets
        self.quadruples = quadruples

        self.forward_asp_query = _forward_asp_query
        self.forward_asp_answer_start = _forward_asp_answer_start
        self.forward_asp_answer_end = _forward_asp_answer_end
        self.forward_opi_query = _forward_opi_query
        self.forward_opi_answer_start = _forward_opi_answer_start
        self.forward_opi_answer_end = _forward_opi_answer_end

        self.forward_asp_query_mask = _forward_asp_query_mask
        self.forward_asp_query_seg = _forward_asp_query_seg
        self.forward_opi_query_mask = _forward_opi_query_mask
        self.forward_opi_query_seg = _forward_opi_query_seg

        self.backward_asp_query = _backward_asp_query
        self.backward_asp_answer_start = _backward_asp_answer_start
        self.backward_asp_answer_end = _backward_asp_answer_end
        self.backward_opi_query = _backward_opi_query
        self.backward_opi_answer_start = _backward_opi_answer_start
        self.backward_opi_answer_end = _backward_opi_answer_end

        self.backward_asp_query_mask = _backward_asp_query_mask
        self.backward_asp_query_seg = _backward_asp_query_seg
        self.backward_opi_query_mask = _backward_opi_query_mask
        self.backward_opi_query_seg = _backward_opi_query_seg

        self.category_query = _category_query
        self.category_answer = _category_answer
        self.category_query_mask = _category_query_mask
        self.category_query_seg = _category_query_seg

        self.sentiment_query = _sentiment_query
        self.sentiment_answer = _sentiment_answer
        self.sentiment_query_mask = _sentiment_query_mask
        self.sentiment_query_seg = _sentiment_query_seg

        self.forward_opi_nums = _forward_opi_nums
        self.backward_asp_nums = _backward_asp_nums
        self.pairs_nums = _pairs_nums

        self.forward_aspect_len = forward_aspect_len
        self.forward_opinion_lens = forward_opinion_lens
        self.backward_opinion_len = backward_opinion_len
        self.backward_aspects_lens = backward_aspects_lens
        self.sentiment_category_lens = sentiment_category_lens