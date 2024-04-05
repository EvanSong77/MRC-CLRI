# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:54
# @Author  : codewen77
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from collate import collate_fn
from data_utils import ACOSDataset
from finetuning_argparse import init_args
from labels import get_aspect_category, get_category_sentiment_num_list
from mrc_model import MRCModel
from tools import get_logger, seed_everything, save_model, print_results, print_results2
from trainer import ACOSTrainer


def do_train():
    # ##########init model##########
    logger.info("Building MRC-CLRI Model...")
    category_list = get_aspect_category(args.task, args.data_type)
    args.category_dim = len(category_list[0])
    # category and sentiment num_list
    res_lists = get_category_sentiment_num_list(args)
    args.category_num_list = res_lists[0]
    args.sentiment_num_list = res_lists[-1]

    model = MRCModel(args, len(category_list[0]))
    model = model.cuda()
    # load dat
    # dataset
    train_dataset = ACOSDataset(tokenizer, args, "train")
    dev_dataset = ACOSDataset(tokenizer, args, "dev")
    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # optimizer
    logger.info('initial optimizer......')
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if "_bert" not in n],
         'lr': args.learning_rate1, 'weight_decay': 0.01}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate2)

    # scheduler
    batch_num_train = len(train_dataset) // args.train_batch_size
    training_steps = args.epoch_num * batch_num_train
    warmup_steps = int(training_steps * args.warm_up)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # trainer
    trainer = ACOSTrainer(logger, model, optimizer, scheduler, tokenizer, args)

    # ##########Training##########
    logger.info("***** Running Training *****")
    best_dev_quadruple_f1, best_test_quadruple_f1, best_test_imp_quad_f1 = .0, .0, .0
    for epoch in range(1, args.epoch_num + 1):
        model.train()
        trainer.train(train_dataloader, epoch)
        # ##########Dev##########
        logger.info("***** Running Dev | Epoch {} *****".format(epoch))
        results = trainer.eval(dev_dataloader)

        if results['quadruple']['f1'] == 0:
            continue
        print_results(logger, results)
        if results['quadruple']['f1'] > best_dev_quadruple_f1:
            best_dev_quadruple_f1 = results['quadruple']['f1']
            save_path = save_model(output_path, f"{args.data_type}_test", epoch, optimizer, model)
            args.save_path = save_path
            logger.info("i got the best dev result {}...".format(best_dev_quadruple_f1))

    logger.info("***** Train Over *****")
    logger.info("The best dev quadruple f1: {}".format(best_dev_quadruple_f1))


def do_test():
    # ##########init model##########
    logger.info("Building MRC-CLRI Model...")
    category_list = get_aspect_category(args.task, args.data_type)
    args.category_dim = len(category_list[0])
    # category and sentiment num_list
    res_lists = get_category_sentiment_num_list(args)
    args.category_num_list = res_lists[0]
    args.sentiment_num_list = res_lists[-1]

    model = MRCModel(args, len(category_list[0]))
    model = model.cuda()
    # load data
    test_dataset = ACOSDataset(tokenizer, args, "test")
    # dataloader
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    # load checkpoint
    if args.save_path:
        checkpoint = torch.load(args.save_path)
    else:
        checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()

    trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
    logger.info("***** Running Test *****")
    # test_results = trainer.eval(test_dataloader)
    test_results = trainer.batch_eval(test_dataloader)

    logger.info(test_results)


def do_eval():
    # ##########init model##########
    logger.info("Building MRC-CLRI Model...")
    category_list = get_aspect_category(args.task, args.data_type)
    args.category_dim = len(category_list[0])
    # category and sentiment num_list
    res_lists = get_category_sentiment_num_list(args)
    args.category_num_list = res_lists[0]
    args.sentiment_num_list = res_lists[-1]

    model = MRCModel(args, len(category_list[0]))

    # load data
    # dataset
    dev_dataset = ACOSDataset(tokenizer, args, "dev")
    test_dataset = ACOSDataset(tokenizer, args, "test")
    # dataloader
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    logger.info('***** Running Testing *****')
    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()

    trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
    # ##########Dev##########
    logger.info("***** Running Dev *****")
    # dev_results = trainer.eval(dev_dataloader)
    dev_results = trainer.batch_eval(dev_dataloader)
    if args.do_optimized:
        print_results2(logger, dev_results)
    else:
        print_results(logger, dev_results)
    logger.info("***** Running Test *****")
    # test_results = trainer.eval(test_dataloader)
    test_results = trainer.batch_eval(test_dataloader)
    if args.do_optimized:
        print_results2(logger, test_results)
    else:
        print_results(logger, test_results)

    return dev_results, test_results


def do_optimized():
    # 先确定beta再确定alpha(alpha=0.8)
    start, end, step = args.alpha_start, args.alpha_end, args.alpha_step
    alpha_list = [round(x, 2) for x in list(np.arange(start, end + step, step))]

    start, end, step = args.beta_start, args.beta_end, args.beta_step
    beta_list = [i for i in range(start, end + 1, step)]

    dev_f1_list, test_f1_list = [], []
    for b in beta_list:
        args.beta = int(b)
        logger.info(f'alpha is {args.alpha}, beta is {b}')
        dev_results, test_results = do_eval()

        dev_f1_list.append(dev_results['quadruple']['f1'])
        test_f1_list.append(test_results['quadruple']['f1'])
    best_dev_f1, best_test_f1 = max(dev_f1_list), max(test_f1_list)
    best_dev_beta, best_test_beta = dev_f1_list.index(best_dev_f1), test_f1_list.index(best_test_f1)
    logger.info(f'The best dev f1:{best_dev_f1}, the best test f1: {best_test_f1}')
    logger.info(dev_f1_list)
    logger.info(f'The best dev beta:{beta_list[best_dev_beta]}, the best test beta: {beta_list[best_test_beta]}')

    # 先确定alpha再确定beta（beta=0）
    dev_f1_list, test_f1_list = [], []
    args.beta = beta_list[best_dev_beta]
    for a in alpha_list:
        args.alpha = a
        logger.info(f'alpha is {a}, beta is {args.beta}')
        dev_results, test_results = do_eval()

        dev_f1_list.append(dev_results['quadruple']['f1'])
        test_f1_list.append(test_results['quadruple']['f1'])
    best_dev_f1, best_test_f1 = max(dev_f1_list), max(test_f1_list)
    best_dev_alpha, best_test_alpha = dev_f1_list.index(best_dev_f1), test_f1_list.index(best_test_f1)
    logger.info(f'The best dev f1:{best_dev_f1}, the best test f1: {best_test_f1}')
    logger.info(dev_f1_list)
    logger.info(f'The best dev alpha:{alpha_list[best_dev_alpha]}, the best test alpha: {alpha_list[best_test_alpha]}')


def do_inference(reviews):
    # ##########init model##########
    logger.info("Building MRC-CLRI Model...")
    category_list = get_aspect_category(args.task, args.data_type)
    args.category_dim = len(category_list[0])
    # category and sentiment num_list
    res_lists = get_category_sentiment_num_list(args)
    args.category_num_list = res_lists[0]
    args.sentiment_num_list = res_lists[-1]

    model = MRCModel(args, len(category_list[0]))
    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()

    # Start do_inference
    print("****** Start do_inference ******")
    trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
    trainer.inference(reviews)
    print()


if __name__ == '__main__':
    # ##########init args##########
    args = init_args()

    # ##########seed##########
    seed_everything(args.seed)

    # ##########创建目录##########
    output_path = os.path.join(args.output_dir, args.task, args.data_type)
    log_path = os.path.join(args.log_dir, args.task, args.data_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # ##########init logger##########
    log_path = os.path.join(log_path, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + ".log")
    logger = get_logger(log_path)

    # print args
    logger.info(args)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    if args.do_train:
        do_train()
    if args.do_test:
        do_test()
    if args.do_optimized:
        do_optimized()
    if args.do_inference:
        # 根据给定的文本直接预测出结果
        text = ['The sushi was awful !',
                # [['sushi', 'FOOD#QUALITY', 'awful', 'negative']]
                'they have a delicious banana chocolate dessert , as well as a great green tea te ##mp ##ura .',
                # [['banana chocolate dessert', 'FOOD#QUALITY', 'delicious', 'positive'], ['green tea te ##mp', 'FOOD#QUALITY', 'great', 'positive']]
                'so delicious ! ! ! ! ! !',
                # [['NULL', 'FOOD#QUALITY', 'delicious', 'positive']]
                ]
        do_inference(text)
