# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:19
# @Author  : codewen77

# Chinese
def get_Chinese_Template():
    Chinese_Forward_Q1 = ["[CLS]", "哪", "些", "方", "面", "？", "[SEP]"]
    Chinese_Backward_Q1 = ["[CLS]", "哪", "些", "意", "见", "？", "[SEP]"]
    Chinese_Forward_Q2 = ["[CLS]", "哪", "些", "意", "见", "对", "于", "？", "[SEP]"]
    Chinese_Backward_Q2 = ["[CLS]", "哪", "些", "方", "面", "对", "于", "？", "[SEP]"]
    Chinese_Q3 = ["[CLS]", "什", "么", "种", "类", "对", "于", "是", "？", "[SEP]"]
    Chinese_Q4 = ["[CLS]", "什", "么", "情", "感", "对", "于", "是", "？", "[SEP]"]

    return Chinese_Forward_Q1, Chinese_Backward_Q1, Chinese_Forward_Q2, Chinese_Backward_Q2, Chinese_Q3, Chinese_Q4


# English
def get_English_Template():
    English_Forward_Q1 = ["[CLS]", "What", "aspects", "?", "[SEP]"]
    English_Backward_Q1 = ["[CLS]", "What", "opinions", "?", "[SEP]"]
    English_Forward_Q2 = ["[CLS]", "What", "opinions", "for", "the", "aspect", "?", "[SEP]"]
    English_Backward_Q2 = ["[CLS]", "What", "aspects", "for", "the", "opinion", "?", "[SEP]"]
    English_Q3 = ["[CLS]", "What", "category", "for", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]
    English_Q4 = ["[CLS]", "What", "sentiment", "for", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]

    return English_Forward_Q1, English_Backward_Q1, English_Forward_Q2, English_Backward_Q2, English_Q3, English_Q4
