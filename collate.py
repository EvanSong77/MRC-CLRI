# -*- coding: utf-8 -*-
# @Time    : 2022/11/6 20:14
# @Author  : codewen77
import torch

from samples import TokenizedSample


def collate_fn(sample):
    sentence_token = [s.sentence_token for s in sample]
    sentence_len = [s.sentence_len for s in sample]

    aspects = [s.aspects for s in sample]
    opinions = [s.opinions for s in sample]
    pairs = [s.pairs for s in sample]
    aste_triplets = [s.aste_triplets for s in sample]
    aoc_triplets = [s.aoc_triplets for s in sample]
    quadruples = [s.quadruples for s in sample]

    forward_asp_query = torch.tensor([s.forward_asp_query for s in sample], dtype=torch.long)
    forward_asp_answer_start = torch.tensor([s.forward_asp_answer_start for s in sample], dtype=torch.long)
    forward_asp_answer_end = torch.tensor([s.forward_asp_answer_end for s in sample], dtype=torch.long)
    forward_asp_query_mask = torch.tensor([s.forward_asp_query_mask for s in sample], dtype=torch.float)
    forward_asp_query_seg = torch.tensor([s.forward_asp_query_seg for s in sample], dtype=torch.long)

    forward_opi_query = torch.tensor([s.forward_opi_query for s in sample], dtype=torch.long)
    forward_opi_answer_start = torch.tensor([s.forward_opi_answer_start for s in sample], dtype=torch.long)
    forward_opi_answer_end = torch.tensor([s.forward_opi_answer_end for s in sample], dtype=torch.long)
    forward_opi_query_mask = torch.tensor([s.forward_opi_query_mask for s in sample], dtype=torch.float)
    forward_opi_query_seg = torch.tensor([s.forward_opi_query_seg for s in sample], dtype=torch.long)

    backward_asp_query = torch.tensor([s.backward_asp_query for s in sample], dtype=torch.long)
    backward_asp_answer_start = torch.tensor([s.backward_asp_answer_start for s in sample], dtype=torch.long)
    backward_asp_answer_end = torch.tensor([s.backward_asp_answer_end for s in sample], dtype=torch.long)
    backward_asp_query_mask = torch.tensor([s.backward_asp_query_mask for s in sample], dtype=torch.float)
    backward_asp_query_seg = torch.tensor([s.backward_asp_query_seg for s in sample], dtype=torch.long)

    backward_opi_query = torch.tensor([s.backward_opi_query for s in sample], dtype=torch.long)
    backward_opi_answer_start = torch.tensor([s.backward_opi_answer_start for s in sample], dtype=torch.long)
    backward_opi_answer_end = torch.tensor([s.backward_opi_answer_end for s in sample], dtype=torch.long)
    backward_opi_query_mask = torch.tensor([s.backward_opi_query_mask for s in sample], dtype=torch.float)
    backward_opi_query_seg = torch.tensor([s.backward_opi_query_seg for s in sample], dtype=torch.long)

    category_query = torch.tensor([s.category_query for s in sample], dtype=torch.long)
    category_answer = torch.tensor([s.category_answer for s in sample], dtype=torch.long)
    category_query_mask = torch.tensor([s.category_query_mask for s in sample], dtype=torch.float)
    category_query_seg = torch.tensor([s.category_query_seg for s in sample], dtype=torch.long)

    sentiment_query = torch.tensor([s.sentiment_query for s in sample], dtype=torch.long)
    sentiment_answer = torch.tensor([s.sentiment_answer for s in sample], dtype=torch.long)
    sentiment_query_mask = torch.tensor([s.sentiment_query_mask for s in sample], dtype=torch.float)
    sentiment_query_seg = torch.tensor([s.sentiment_query_seg for s in sample], dtype=torch.long)

    forward_opi_nums = [s.forward_opi_nums for s in sample]
    backward_asp_nums = [s.backward_asp_nums for s in sample]
    pairs_nums = [s.pairs_nums for s in sample]

    forward_aspect_len = [s.forward_aspect_len for s in sample]
    forward_opinion_lens = [s.forward_opinion_lens for s in sample]
    backward_opinion_len = [s.backward_opinion_len for s in sample]
    backward_aspects_lens = [s.backward_aspects_lens for s in sample]
    sentiment_category_lens = [s.sentiment_category_lens for s in sample]

    return TokenizedSample(sentence_token, sentence_len,
                           aspects, opinions, pairs, aste_triplets, aoc_triplets, quadruples,
                           forward_asp_query, forward_asp_answer_start, forward_asp_answer_end, forward_asp_query_mask, forward_asp_query_seg,
                           forward_opi_query, forward_opi_answer_start, forward_opi_answer_end, forward_opi_query_mask, forward_opi_query_seg,
                           backward_asp_query, backward_asp_answer_start, backward_asp_answer_end, backward_asp_query_mask, backward_asp_query_seg,
                           backward_opi_query, backward_opi_answer_start, backward_opi_answer_end, backward_opi_query_mask, backward_opi_query_seg,
                           category_query, category_answer, category_query_mask, category_query_seg,
                           sentiment_query, sentiment_answer, sentiment_query_mask, sentiment_query_seg,
                           forward_opi_nums, backward_asp_nums, pairs_nums,
                           forward_aspect_len, forward_opinion_lens, backward_opinion_len, backward_aspects_lens, sentiment_category_lens)