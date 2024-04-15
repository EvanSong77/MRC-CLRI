# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:43
# @Author  : codewen77
import argparse


def init_args():
    parser = argparse.ArgumentParser()
    # Base parameters
    parser.add_argument("--data_path", default='', type=str, required=True, help="The path of data")
    parser.add_argument("--task", default='ASQP', type=str, required=True,
                        help="The name of the task, selected from: [ACOS, ASQP]")
    parser.add_argument("--data_type", default='rest15', type=str, required=True,
                        help="The type of the data, selected from: [laptop, rest], [rest15, rest16]")
    parser.add_argument("--model_path", default='', required=True, type=str, help="Path to pre-trained model")

    parser.add_argument("--output_dir", default='./outputs/saves', type=str, help="The dir of results")
    parser.add_argument('--log_dir', default="./outputs/logs", type=str, help="The dir of logs")
    parser.add_argument('--save_path', default="", type=str)
    parser.add_argument('--checkpoint_path', default="./outputs/saves/ASQE/review/1668339564_modeltest.pth", type=str,
                        help="The dir of checkpoint")

    # 训练与测试
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_optimized", action='store_true')
    parser.add_argument("--do_inference", action='store_true')
    # 对比学习
    parser.add_argument("--use_category_SCL", action='store_true', help="Use Contrastive Learning Loss")
    parser.add_argument("--use_sentiment_SCL", action='store_true', help="Use Contrastive Learning Loss")
    # 对抗训练
    parser.add_argument("--use_FGM", action='store_true', help="Use Confrontation Training")
    parser.add_argument("--use_PGD", action='store_true', help="Use Confrontation Training")
    # loss
    parser.add_argument("--use_FocalLoss", action='store_true', help="Use Focal Loss")
    parser.add_argument("--use_LDAMLoss", action='store_true', help="Use LDAM Loss")
    parser.add_argument("--use_LMFLoss", action='store_true', help="Use LMF Loss")
    # w/o forward or backward
    parser.add_argument("--use_Forward", action='store_true', help="Only use forward mrc")
    parser.add_argument("--use_Backward", action='store_true', help="Only use backward mrc")

    # Other parameters
    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--learning_rate1", default=1e-3, type=float)
    parser.add_argument("--learning_rate2", default=3e-5, type=float)
    parser.add_argument("--epoch_num", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--low_resource', type=float, default=1.0)

    # contrastive hyper parameters
    # category
    parser.add_argument('--contrastive_lr1', default=1e-4, type=float)
    # sentiment
    parser.add_argument('--contrastive_lr2', default=1e-5, type=float)

    # pgd hyper parameters
    parser.add_argument("--pgd_k", default=3, type=int)

    # Focal Loss parameters
    parser.add_argument('--flp_gamma', type=float, default=2)

    # optimized parameters
    parser.add_argument('--alpha_start', type=float, default=0.80)
    parser.add_argument('--alpha_end', type=float, default=1.0)
    parser.add_argument('--alpha_step', type=float, default=0.02)
    parser.add_argument('--beta_start', type=int, default=20)
    parser.add_argument('--beta_end', type=int, default=70)
    parser.add_argument('--beta_step', type=int, default=5)
    parser.add_argument('--beta_list', type=list, default=[])
    parser.add_argument('--alpha', type=float, default=0.80)
    parser.add_argument('--beta', type=int, default=0)

    args = parser.parse_args()

    return args
