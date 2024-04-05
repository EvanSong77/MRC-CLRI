import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from data_utils import ACOSDataset
from mrc_model import MRCModel
from trainer import ACOSTrainer
from collate import collate_fn
from finetuning_argparse import init_args
from labels import get_aspect_category, get_category_sentiment_num_list
from tools import seed_everything, get_logger

# ##########init args##########
args = init_args()

#
args.beta, args.alpha = 25, 0.82

# ##########seed##########
seed_everything(args.seed)
logger = get_logger('./test.log')

category_list = get_aspect_category(args.task, args.data_type)
tokenizer = BertTokenizer.from_pretrained(args.model_path)

# category and sentiment num_list
res_lists = get_category_sentiment_num_list(args)
args.category_num_list = res_lists[0]
args.sentiment_num_list = res_lists[-1]
model = MRCModel(args, len(category_list[0]))
model = model.cuda()

test_dataset = ACOSDataset(tokenizer, args, "test")
# dataloader
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
logger.info('***** Running Testing *****')
print(args)
# load checkpoint
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['net'])
model = model.cuda()

trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
test_results = trainer.test(test_dataloader)
print(test_results)
"""
nohup python run.py \
    --data_path ./data/QUAD/v2/rest15/ \
    --task QUAD \
    --data_type rest15 \
    --model_path /home/codewen/codewen_workspace/pretrained-models/SentiWSP \
    --checkpoint_path /home/codewen/codewen_workspace/MRC-RICL/outputs/saves/QUAD/rest15/rest15_test_model.pth \
    --do_eval
"""