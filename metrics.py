# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 13:42
# @Author  : codewen77


class ACOSScore(object):
    """
    指标计算:
    aspect,
    opinion,
    (aspect,opinion) pair,
    (aspect,opinion,sentiment) triplet,
    (aspect,category,opinion,sentiemnt) quadruple
    imp_quadruple
    """

    def __init__(self, logger):
        self.logger = logger
        # aspect
        self.true_asp = .0
        self.pred_asp = .0
        self.gold_asp = .0
        # opinion
        self.true_opi = .0
        self.pred_opi = .0
        self.gold_opi = .0
        # (aspect, opinion) pair
        self.true_ao_pair = .0
        self.pred_ao_pair = .0
        self.gold_ao_pair = .0
        # (aspect, opinion, sentiment) triplet
        self.true_aste_triplet = .0
        self.pred_aste_triplet = .0
        self.gold_aste_triplet = .0
        # (aspect, opinion, category) aoc triplet
        self.true_aoc_triplet = .0
        self.pred_aoc_triplet = .0
        self.gold_aoc_triplet = .0
        # imp_quadruple
        self.true_imp_quadruple = .0
        self.pred_imp_quadruple = .0
        self.gold_imp_quadruple = .0
        # (aspect, category, opinion, sentiemnt) quadruple
        self.true_quadruple = .0
        self.pred_quadruple = .0
        self.gold_quadruple = .0

    def compute(self):
        # aspect
        asp_p = 0 if self.true_asp + self.pred_asp == 0 else 1. * self.true_asp / self.pred_asp
        asp_r = 0 if self.true_asp + self.gold_asp == 0 else 1. * self.true_asp / self.gold_asp
        asp_f = 0 if asp_p + asp_r == 0 else 2 * asp_p * asp_r / (asp_p + asp_r)
        # self.logger.info({"aspect": {"true_asp": self.true_asp, "pred_asp": self.pred_asp, "gold_asp": self.gold_asp}})
        # opinion
        opi_p = 0 if self.true_opi + self.pred_opi == 0 else 1. * self.true_opi / self.pred_opi
        opi_r = 0 if self.true_opi + self.gold_opi == 0 else 1. * self.true_opi / self.gold_opi
        opi_f = 0 if opi_p + opi_r == 0 else 2 * opi_p * opi_r / (opi_p + opi_r)
        # self.logger.info({"opinion": {"true_opi": self.true_opi, "pred_opi": self.pred_opi, "gold_opi": self.gold_opi}})
        # (aspect, opinion) pair
        ao_pair_p = 0 if self.true_ao_pair + self.pred_ao_pair == 0 else 1. * self.true_ao_pair / self.pred_ao_pair
        ao_pair_r = 0 if self.true_ao_pair + self.gold_ao_pair == 0 else 1. * self.true_ao_pair / self.gold_ao_pair
        ao_pair_f = 0 if ao_pair_p + ao_pair_r == 0 else 2 * ao_pair_p * ao_pair_r / (ao_pair_p + ao_pair_r)
        # self.logger.info({"ao_pair": {"true_ao_pair": self.true_ao_pair, "pred_ao_pair": self.pred_ao_pair, "gold_ao_pair": self.gold_ao_pair}})
        # (aspect, opinion, sentiment) triplet
        aste_triplet_p = 0 if self.true_aste_triplet + self.pred_aste_triplet == 0 else 1. * self.true_aste_triplet / self.pred_aste_triplet
        aste_triplet_r = 0 if self.true_aste_triplet + self.gold_aste_triplet == 0 else 1. * self.true_aste_triplet / self.gold_aste_triplet
        aste_triplet_f = 0 if aste_triplet_p + aste_triplet_r == 0 else 2 * aste_triplet_p * aste_triplet_r / (
                aste_triplet_p + aste_triplet_r)
        # self.logger.info({"aste_triplet": {"true_aste_triplet": self.true_aste_triplet, "pred_aste_triplet": self.pred_aste_triplet, "gold_aste_triplet": self.gold_aste_triplet}})
        # aoc tirplet
        aoc_triplet_p = 0 if self.true_aoc_triplet + self.pred_aoc_triplet == 0 else 1. * self.true_aoc_triplet / self.pred_aoc_triplet
        aoc_triplet_r = 0 if self.true_aoc_triplet + self.gold_aoc_triplet == 0 else 1. * self.true_aoc_triplet / self.gold_aoc_triplet
        aoc_triplet_f = 0 if aoc_triplet_p + aoc_triplet_r == 0 else 2 * aoc_triplet_p * aoc_triplet_r / (
                aoc_triplet_p + aoc_triplet_r)
        # self.logger.info({"aoc_triplet": {"true_aoc_triplet": self.true_aoc_triplet, "pred_aoc_triplet": self.pred_aoc_triplet, "gold_aoc_triplet": self.gold_aoc_triplet}})

        # imp_quadruple
        imp_quadruple_p = 0 if self.true_imp_quadruple + self.pred_imp_quadruple == 0 else 1. * self.true_imp_quadruple / self.pred_imp_quadruple
        imp_quadruple_r = 0 if self.true_imp_quadruple + self.gold_imp_quadruple == 0 else 1. * self.true_imp_quadruple / self.gold_imp_quadruple
        imp_quadruple_f = 0 if imp_quadruple_p + imp_quadruple_r == 0 else 2 * imp_quadruple_p * imp_quadruple_r / (
                imp_quadruple_p + imp_quadruple_r)
        # self.logger.info({"imp_quadruple": {"true_imp_quadruple": self.true_imp_quadruple, "pred_imp_quadruple": self.pred_imp_quadruple, "gold_imp_quadruple": self.gold_imp_quadruple}})
        # (aspect, category, opinion, sentiment) quadruple
        quadruple_p = 0 if self.true_quadruple + self.pred_quadruple == 0 else 1. * self.true_quadruple / self.pred_quadruple
        quadruple_r = 0 if self.true_quadruple + self.gold_quadruple == 0 else 1. * self.true_quadruple / self.gold_quadruple
        quadruple_f = 0 if quadruple_p + quadruple_r == 0 else 2 * quadruple_p * quadruple_r / (
                quadruple_p + quadruple_r)
        # self.logger.info({"quadruple": {"true_quadruple": self.true_quadruple, "pred_quadruple": self.pred_quadruple, "gold_quadruple": self.gold_quadruple}})

        return {"aspect": {"precision": asp_p, "recall": asp_r, "f1": asp_f},
                "opinion": {"precision": opi_p, "recall": opi_r, "f1": opi_f},
                "ao_pair": {"precision": ao_pair_p, "recall": ao_pair_r, "f1": ao_pair_f},
                "aste_triplet": {"precision": aste_triplet_p, "recall": aste_triplet_r, "f1": aste_triplet_f},
                "aoc_triplet": {"precision": aoc_triplet_p, "recall": aoc_triplet_r, "f1": aoc_triplet_f},
                "imp_quadruple": {"precision": imp_quadruple_p, "recall": imp_quadruple_r, "f1": imp_quadruple_f},
                "quadruple": {"precision": quadruple_p, "recall": quadruple_r, "f1": quadruple_f}
                }

    def compute2(self):
        # (aspect, category, opinion, sentiment) quadruple
        quadruple_p = 0 if self.true_quadruple + self.pred_quadruple == 0 else 1. * self.true_quadruple / self.pred_quadruple
        quadruple_r = 0 if self.true_quadruple + self.gold_quadruple == 0 else 1. * self.true_quadruple / self.gold_quadruple
        quadruple_f = 0 if quadruple_p + quadruple_r == 0 else 2 * quadruple_p * quadruple_r / (
                quadruple_p + quadruple_r)

        return {"quadruple": {"precision": quadruple_p, "recall": quadruple_r, "f1": quadruple_f}}

    def update(self, gold_aspects, gold_opinions, gold_ao_pairs, gold_aste_triplets, gold_aoc_triplets, gold_quadruples,
               pred_aspects, pred_opinions, pred_ao_pairs, pred_aste_triplets, pred_aoc_triplets, pred_quadruples):

        self.gold_asp += len(gold_aspects)
        self.gold_opi += len(gold_opinions)
        self.gold_ao_pair += len(gold_ao_pairs)
        self.gold_aste_triplet += len(gold_aste_triplets)
        self.gold_aoc_triplet += len(gold_aoc_triplets)
        self.gold_quadruple += len(gold_quadruples)

        self.pred_asp += len(pred_aspects)
        self.pred_opi += len(pred_opinions)
        self.pred_ao_pair += len(pred_ao_pairs)
        self.pred_aste_triplet += len(pred_aste_triplets)
        self.pred_aoc_triplet += len(pred_aoc_triplets)
        self.pred_quadruple += len(pred_quadruples)

        for g in gold_aspects:
            for p in pred_aspects:
                if g == p:
                    self.true_asp += 1

        for g in gold_opinions:
            for p in pred_opinions:
                if g == p:
                    self.true_opi += 1

        for g in gold_ao_pairs:
            for p in pred_ao_pairs:
                if g == p:
                    self.true_ao_pair += 1

        for g in gold_aste_triplets:
            for p in pred_aste_triplets:
                if g == p:
                    self.true_aste_triplet += 1

        for g in gold_aoc_triplets:
            for p in pred_aoc_triplets:
                if g == p:
                    self.true_aoc_triplet += 1

        for g in gold_quadruples:
            for p in pred_quadruples:
                if g == p:
                    self.true_quadruple += 1

        # 隐式四元组
        imp_gold_quad, imp_pred_quad = [], []
        for g in gold_quadruples:
            if g[0] == [-1, -1] or g[2] == [-1, -1]:
                imp_gold_quad.append(g)
        for p in pred_quadruples:
            if p[0] == [-1, -1] or p[2] == [-1, -1]:
                imp_pred_quad.append(p)
        self.gold_imp_quadruple += len(imp_gold_quad)
        self.pred_imp_quadruple += len(imp_pred_quad)
        for g in imp_gold_quad:
            for p in imp_pred_quad:
                if g == p:
                    self.true_imp_quadruple += 1

    def update2(self, gold_quadruples, pred_quadruples):
        for b in range(len(gold_quadruples)):
            self.gold_quadruple += len(gold_quadruples[b])
            self.pred_quadruple += len(pred_quadruples[b])
            for g in gold_quadruples[b]:
                for p in pred_quadruples[b]:
                    if g == p:
                        self.true_quadruple += 1
