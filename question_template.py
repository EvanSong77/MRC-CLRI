# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:19
# @Author  : codewen77

# English
def get_English_Template():
    English_Forward_Q1 = ["[CLS]", "What", "aspects", "?", "[SEP]"]
    English_Backward_Q1 = ["[CLS]", "What", "opinions", "?", "[SEP]"]
    English_Forward_Q2 = ["[CLS]", "What", "opinions", "for", "the", "aspect", "?", "[SEP]"]
    English_Backward_Q2 = ["[CLS]", "What", "aspects", "for", "the", "opinion", "?", "[SEP]"]
    English_Q3 = ["[CLS]", "What", "category", "for", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]
    English_Q4 = ["[CLS]", "What", "sentiment", "for", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]

    return English_Forward_Q1, English_Backward_Q1, English_Forward_Q2, English_Backward_Q2, English_Q3, English_Q4
