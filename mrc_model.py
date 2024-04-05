# -*- coding: utf-8 -*-
# @Time    : 2022/11/6 9:38
# @Author  : codewen77
import torch.nn as nn
from transformers import BertModel, RobertaModel, RobertaConfig


class MRCModel(nn.Module):
    def __init__(self, args, category_dim):
        # hidden_size = BertConfig.from_pretrained(args.model_path).hidden_size
        hidden_size = RobertaConfig.from_pretrained(args.model_path).hidden_size
        super(MRCModel, self).__init__()

        # BERT或者Robert模型
        if hidden_size == 768:
            self._bert = BertModel.from_pretrained(args.model_path)
        else:
            if 'SentiWSP' in args.model_path or hidden_size == 1024:
                self._bert = BertModel.from_pretrained(args.model_path)
            else:
                self._bert = RobertaModel.from_pretrained(args.model_path)
        # self._bert = RobertaModel.from_pretrained(args.model_path)

        # a o实体分类器
        self.classifier_start = nn.Linear(hidden_size, 2)
        self.classifier_end = nn.Linear(hidden_size, 2)

        # 类别分类器
        self._classifier_category = nn.Linear(hidden_size, category_dim)
        # 情感分类器
        self._classifier_sentiment = nn.Linear(hidden_size, 3)

    def forward(self, query_tensor, query_mask, query_seg, step, inputs_embeds=None):
        # hidden_states = self._bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg, inputs_embeds=inputs_embeds)[0]
        hidden_states = self._bert(query_tensor, attention_mask=query_mask)[0]
        if step == 0:  # predict entity
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            return out_scores_start, out_scores_end
        elif step == 1:  # predict category
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_category(cls_hidden_states)
            return cls_hidden_scores
        else:  # predict sentiment
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_sentiment(cls_hidden_states)
            return cls_hidden_scores
