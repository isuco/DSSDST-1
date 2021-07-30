# coding = utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils import helper
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm
from models.model import DualReader
import torch.nn as nn
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils import constant
from utils.fix_label import fix_general_label_error
# from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
# from utils.ckpt_utils import download_ckpt, convert_ckpt_compatible
# from evaluation import model_evaluation
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from evaluation import model_evaluation,op_evaluation
from transformers.configuration_albert import AlbertConfig
from transformers.tokenization_albert import AlbertTokenizer
from transformer import AdamW
from pytorch_transformers import AdamW, WarmupLinearSchedule

import sys
import csv

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

track_slots = ["attraction-area",
               "attraction-name",
               "attraction-type",
               "bus-day",
               "bus-departure",
               "bus-destination",
               "bus-leaveat",
               "hospital-department",
               "hotel-area",
               "hotel-bookday",
               "hotel-bookpeople",
               "hotel-bookstay",
               "hotel-internet",
               "hotel-name",
               "hotel-parking",
               "hotel-pricerange",
               "hotel-stars",
               "hotel-type",
               "restaurant-area",
               "restaurant-bookday",
               "restaurant-bookpeople",
               "restaurant-booktime",
               "restaurant-food",
               "restaurant-name",
               "restaurant-pricerange",
               "taxi-arriveby",
               "taxi-departure",
               "taxi-destination",
               "taxi-leaveat",
               "train-arriveby",
               "train-bookpeople",
               "train-day",
               "train-departure",
               "train-destination",
               "train-leaveat"]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

'''compute joint operation scores based on logits of two stages 
'''
def compute_jointscore(start_scores, end_scores, gen_scores, pred_ops, ans_vocab, slot_mask):
    seq_lens = start_scores.shape[-1]
    joint_score = start_scores.unsqueeze(-2) + end_scores.unsqueeze(-1)
    triu_mask = np.triu(np.ones((joint_score.size(-1), joint_score.size(-1))))
    triu_mask[0, 1:] = 0
    triu_mask = (torch.Tensor(triu_mask) ==  0).bool()
    joint_score = joint_score.masked_fill(triu_mask.unsqueeze(0).unsqueeze(0).cuda(),-1e9).masked_fill(
        slot_mask.unsqueeze(1).unsqueeze(-2)  == 0, -1e9)
    joint_score = F.softmax(joint_score.view(joint_score.size(0), joint_score.size(1), -1),
                            dim = -1).view(joint_score.size(0), joint_score.size(1), seq_lens, -1)

    score_diff = (joint_score[:, :, 0, 0] - joint_score[:, :, 1:, 1:].max(dim = -1)[0].max(dim = -1)[
        0])
    score_noans = pred_ops[:, :, -1] - pred_ops[:, :, 0]
    slot_ans_count = (ans_vocab.sum(-1) != 0).sum(dim=-1)-2
    ans_idx = torch.where(slot_ans_count < 0, torch.zeros_like(slot_ans_count), slot_ans_count)
    neg_ans_mask = torch.cat((torch.linspace(0, ans_vocab.size(0) - 1,
                                             ans_vocab.size(0)).unsqueeze(0).long(),
                              ans_idx.unsqueeze(0)),
                             dim = 0)
    neg_ans_mask = torch.sparse_coo_tensor(neg_ans_mask, torch.ones(ans_vocab.size(0)),
                                           (ans_vocab.size(0),
                                            ans_vocab.size(1))).to_dense().cuda()
    score_neg = gen_scores.masked_fill(neg_ans_mask.unsqueeze(0) == 0, -1e9).max(dim=-1)[0]
    score_has = gen_scores.masked_fill(neg_ans_mask.unsqueeze(0) == 1, -1e9).max(dim=-1)[0]
    cate_score_diff = score_neg - score_has
    score_diffs = score_diff.view(-1).cpu().detach().numpy().tolist()
    cate_score_diffs = cate_score_diff.view(-1).cpu().detach().numpy().tolist()
    score_noanses = score_noans.view(-1).cpu().detach().numpy().tolist()
    return score_diffs, cate_score_diffs, score_noanses


def saveOperationLogits(model, device, dataset, save_path, turn):
    score_ext_map = {}
    model.eval()
    for batch in tqdm(dataset, desc = "Evaluating"):
        batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in batch]
        input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, sid = batch
        batch_size = input_ids.shape[0]
        seq_lens = input_ids.shape[1]
        start_logits, end_logits, has_ans, gen_scores, _, _, _ = model(input_ids = input_ids,
                                                                       token_type_ids = segment_ids,
                                                                       state_positions = state_position_ids,
                                                                       attention_mask = input_mask,
                                                                       slot_mask = slot_mask,
                                                                       max_value = max_value,
                                                                       op_ids = op_ids,
                                                                       max_update = max_update)

        score_ext = has_ans.cpu().detach().numpy().tolist()
        for i, sd in enumerate(score_ext):
            score_ext_map[sid[i]] = sd
    with open(os.path.join(save_path, "cls_score_test_turn{}.json".format(turn)), "w") as writer:
        writer.write(json.dumps(score_ext_map, indent = 4) + "\n")


def masked_cross_entropy_for_value(logits, target, sample_mask = None, slot_mask = None, pad_idx = -1):
    mask = logits.eq(0)
    pad_mask = target.ne(pad_idx)
    target = target.masked_fill(target < 0, 0)
    sample_mask = pad_mask & sample_mask if sample_mask is not None else pad_mask
    sample_mask = slot_mask & sample_mask if slot_mask is not None else sample_mask
    target = target.masked_fill(sample_mask ==  0, 0)
    logits = logits.masked_fill(mask, 1)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim = 1, index = target_flat)
    losses = losses_flat.view(*target.size())
    # if mask is not None:
    sample_num = sample_mask.sum().float()
    losses = losses * sample_mask.float()
    loss = (losses.sum() / sample_num) if sample_num != 0 else losses.sum()
    return loss

# [SLOT], [NULL], [EOS]
def addSpecialTokens(tokenizer, specialtokens):
    special_key = "additional_special_tokens"
    tokenizer.add_special_tokens({special_key: specialtokens})

def fixontology(ontology, turn, tokenizer):
    ans_vocab = []
    esm_ans_vocab = []
    esm_ans = constant.ansvocab
    slot_map = constant.slot_map
    slot_mm = np.zeros((len(slot_map), len(esm_ans)))
    max_anses_length = 0
    max_anses = 0
    for i, k in enumerate(ontology.keys()):
        if k in track_slots:
            s = ontology[k]
            s['name'] = k
            if not s['type']:
                s['db'] = []
            slot_mm[i][slot_map[s['name']]] = 1
            ans_vocab.append(s)
    for si in esm_ans:
        slot_anses = []
        for ans in si:
            enc_ans=tokenizer.encode(ans)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(enc_ans)
        max_anses = max(max_anses, len(slot_anses))
        esm_ans_vocab.append(slot_anses)
    for s in esm_ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    esm_ans_vocab = np.array(esm_ans_vocab)
    ans_vocab_tensor = torch.from_numpy(esm_ans_vocab)
    slot_mm = torch.from_numpy(slot_mm).float()
    return ans_vocab, slot_mm, ans_vocab_tensor


def mask_ans_vocab(ontology, slot_meta, tokenizer):
    ans_vocab = []
    max_anses = 0
    max_anses_length = 0
    change_k = []
    cate_mask = []
    for k in ontology:
        if (' range' in k['name']) or (' at' in k['name']) or (' by' in k['name']):
            change_k.append(k)
        # fix_label(ontology[k])
    for key in change_k:
        new_k = key['name'].replace(' ', '')
        key['name'] = new_k
    for s in ontology:
        cate_mask.append(s['type'])
        v_list = s['db']
        slot_anses = []
        for v in v_list:
            ans = tokenizer.encode(v)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(ans)
        max_anses = max(max_anses, len(slot_anses))
        ans_vocab.append(slot_anses)
    for s in ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    ans_vocab = np.array(ans_vocab)
    ans_vocab_tensor = torch.from_numpy(ans_vocab)
    return ans_vocab_tensor, ans_vocab, cate_mask


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type", default = 'albert', type = str,
                        help = "Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default = 'pretrained_models/albert_large/', type = str,
                        help = "Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default = "saved_models/", type = str,
                        help = "The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--config_name", default = "", type = str,
                        help = "Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default = "", type = str,
                        help = "Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default = "", type = str,
                        help = "Where do you want to store the pre-trained models downloaded from s3")
    # parser.add_argument("--max_seq_length", default = 128, type = int,
    #                     help = "The maximum total input sequence length after tokenization. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", default=False, action = 'store_true',
                        help = "Whether to run training.")
    parser.add_argument("--evaluate_during_training", action = 'store_true',
                        help = "Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action = 'store_true',
                        help = "Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default = 32, type = int,
                        help = "Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default = 32, type = int,
                        help = "Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default = 5e-5, type = float,
                        help = "The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default = 0.1, type = float,
                        help = "Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default = 1e-8, type = float,
                        help = "Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help = "Max gradient norm.")
    parser.add_argument("--num_train_epochs", default = 3.0, type = float,
                        help = "Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default = -1, type = int,
                        help = "If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default = 0, type = int,
                        help = "Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type = int, default = 50,
                        help = "Log every X updates steps.")
    parser.add_argument('--save_steps', type = int, default = 50,
                        help = "Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action = 'store_true',
                        help = "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action = 'store_true',
                        help = "Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default = True, action = 'store_true',
                        help = "Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action = 'store_true',
                        help = "Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type = int, default = 42,
                        help = "random seed for initialization")

    parser.add_argument('--fp16', action = 'store_true',
                        help = "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type = str, default = 'O1',
                        help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--local_rank", type = int, default = -1,
                        help = "For distributed training: local_rank")  # DST params
    parser.add_argument("--data_root", default = 'data/mwz2.2/', type = str)
    parser.add_argument("--train_data", default = 'train_dials.json', type = str)
    parser.add_argument("--dev_data", default = 'dev_dials.json', type = str)
    parser.add_argument("--test_data", default = 'test_dials.json', type = str)
    parser.add_argument("--ontology_data", default = 'schema.json', type = str)
    parser.add_argument("--vocab_path", default = 'assets/vocab.txt', type = str)
    parser.add_argument("--save_dir", default = 'saved_models', type = str)
    parser.add_argument("--load_model", default=False, action = 'store_true')
    parser.add_argument("--random_seed", default = 42, type = int)
    parser.add_argument("--num_workers", default = 4, type = int)
    parser.add_argument("--batch_size", default = 8, type = int)
    parser.add_argument("--enc_warmup", default = 0.01, type = float)
    parser.add_argument("--dec_warmup", default = 0.01, type = float)
    parser.add_argument("--enc_lr", default = 5e-6, type = float)
    parser.add_argument("--base_lr", default = 3e-2, type = float)
    parser.add_argument("--n_epochs", default = 10, type = int)
    parser.add_argument("--eval_epoch", default = 1, type = int)

    parser.add_argument("--turn", default = 0, type = int)
    parser.add_argument("--op_code", default = "2", type = str)
    parser.add_argument("--slot_token", default = "[SLOT]", type = str)
    parser.add_argument("--dropout", default = 0.0, type = float)
    parser.add_argument("--hidden_dropout_prob", default = 0.0, type = float)
    parser.add_argument("--attention_probs_dropout_prob", default = 0.1, type = float)
    parser.add_argument("--decoder_teacher_forcing", default = 0.5, type = float)
    parser.add_argument("--word_dropout", default = 0.1, type = float)
    parser.add_argument("--not_shuffle_state", default = True, action = 'store_true')

    parser.add_argument("--n_history", default = 1, type = int)
    parser.add_argument("--max_seq_length", default = 256, type = int)
    parser.add_argument("--sketch_weight", default = 0.55, type = float)
    parser.add_argument("--answer_weight", default = 0.6, type = float)
    parser.add_argument("--generation_weight", default = 0.2, type = float)
    parser.add_argument("--extraction_weight", default = 0.1, type = float)
    parser.add_argument("--msg", default = None, type = str)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    # Prepare GLUE task
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    turn = args.turn

    ontology = json.load(open(args.data_root + args.ontology_data))

    _, slot_meta = make_slot_meta(ontology)
    with torch.cuda.device(0):
        op2id = OP_SET[args.op_code]
        rng = random.Random(args.random_seed)
        print(op2id)
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path + "spiece.model")
        addSpecialTokens(tokenizer, ['[SLOT]', '[NULL]', '[EOS]', '[dontcare]', '[negans]', '[noans]'])
        args.vocab_size = len(tokenizer)
        ontology, slot_mm, esm_ans_vocab = fixontology(slot_meta, turn, tokenizer)
        ans_vocab, ans_vocab_nd, cate_mask = mask_ans_vocab(ontology, slot_meta, tokenizer)
        model = DualReader(args, len(op2id), len(domain2id), op2id['update'], esm_ans_vocab, slot_mm, turn = turn)
        if turn > 0:
            train_op_data_path = args.data_root + "cls_score_train_turn0.json"
            test_op_data_path = args.data_root + "cls_score_test_turn0.json"
            isfilter = True
        else:
            train_op_data_path = None
            test_op_data_path = None
            isfilter = False
        train_data_raw, _, _ = prepare_dataset(data_path = args.data_root + args.train_data,
                                               tokenizer = tokenizer,
                                               slot_meta = slot_meta,
                                               n_history = args.n_history,
                                               max_seq_length = args.max_seq_length,
                                               op_code = args.op_code,
                                               slot_ans = ontology,
                                               turn = turn,
                                               op_data_path = train_op_data_path,
                                               isfilter = isfilter
                                               )

        train_data = MultiWozDataset(train_data_raw,
                                     tokenizer,
                                     slot_meta,
                                     args.max_seq_length,
                                     ontology,
                                     args.word_dropout,
                                     turn = turn)
        print("# train examples %d" % len(train_data_raw))

        dev_data_raw, idmap, _ = prepare_dataset(data_path = args.data_root + args.dev_data,
                                                 tokenizer = tokenizer,
                                                 slot_meta = slot_meta,
                                                 n_history = args.n_history,
                                                 max_seq_length = args.max_seq_length,
                                                 op_code = args.op_code,
                                                 turn = turn,
                                                 slot_ans = ontology,
                                                 op_data_path = test_op_data_path,
                                                 isfilter = False)

        print("# dev examples %d" % len(dev_data_raw))
        dev_data = MultiWozDataset(dev_data_raw,
                                   tokenizer,
                                   slot_meta,
                                   args.max_seq_length,
                                   ontology,
                                   word_dropout = 0,
                                   turn = turn)

        test_data_raw, idmap, _ = prepare_dataset(data_path=args.data_root + args.test_data,
                                                 tokenizer=tokenizer,
                                                 slot_meta=slot_meta,
                                                 n_history=args.n_history,
                                                 max_seq_length=args.max_seq_length,
                                                 op_code=args.op_code,
                                                 turn=turn,
                                                 slot_ans=ontology,
                                                 op_data_path=test_op_data_path,
                                                 isfilter=False)
        # op_data_path=args.data_root+"cls_score_test.json",

        print("# test examples %d" % len(test_data_raw))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler = train_sampler,
                                      batch_size = args.batch_size,
                                      collate_fn = train_data.collate_fn,
                                      num_workers = args.num_workers,
                                      worker_init_fn = worker_init_fn)
        dev_sampler = RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    sampler = dev_sampler,
                                    batch_size = args.batch_size,
                                    collate_fn = dev_data.collate_fn,
                                    num_workers = args.num_workers,
                                    worker_init_fn = worker_init_fn)
        if args.load_model:
            checkpoint = torch.load(os.path.join(args.save_dir, 'model_best_turn{}.bin'.format(turn)))
            # parameters_require = [par[0] for par in model.named_parameters()]
            # parameters_map = OrderedDict()
            # for key in checkpoint['model'].keys():
            #     if key in parameters_require or key[:6] == 'albert':
            #         parameters_map[key] = checkpoint['model'][key]
            # model.load_state_dict(parameters_map)
            model.load_state_dict(checkpoint['model'])
        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)
        if args.do_train:
            num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
            bert_params_ids = list(map(id, model.albert.parameters()))
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            enc_param_optimizer = list(model.named_parameters())

            enc_optimizer_grouped_parameters = [
                {'params': [p for n, p in enc_param_optimizer if
                            (id(p) in bert_params_ids and not any(nd in n for nd in no_decay))], 'weight_decay': 0.01,
                 'lr': args.enc_lr},
                {'params': [p for n, p in enc_param_optimizer if
                            id(p) in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.enc_lr},
                {'params': [p for n, p in enc_param_optimizer if
                            id(p) not in bert_params_ids and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.base_lr},
                {'params': [p for n, p in enc_param_optimizer if
                            id(p) not in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.base_lr}]

            enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr = args.base_lr)

            enc_scheduler = WarmupLinearSchedule(enc_optimizer, int(num_train_steps * args.enc_warmup),
                                                 t_total = num_train_steps)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model, device_ids = [0, 1])

            loss_fnc = nn.CrossEntropyLoss(reduction = 'mean')
            best_score = {'epoch': 0, 'gen_acc': 0, 'op_acc': 0, 'op_F1': 0}
            file_logger = helper.FileLogger(args.save_dir + '/log.txt',
                                            header = "# epoch\ttrain_loss\tdev_gscore\tdev_oscore\tdev_opp\tdev_opr\tdev_f1\tbest_gscore\tbest_oscore\tbest_opf1")
            model.train()
            loss = 0
            sketchy_weight, answer_weight, generation_weight, extraction_weight = args.sketch_weight, args.answer_weight, args.generation_weight, args.extraction_weight
            verify_weight = 1 - sketchy_weight
            for epoch in range(args.n_epochs):
                batch_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in batch]
                    input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, sid = batch
                    batch_cate_mask = torch.BoolTensor(cate_mask).unsqueeze(0).repeat(input_ids.shape[0], 1).cuda()

                    start_logits, end_logits, has_ans, gen_scores, _, _, _ = model(input_ids = input_ids,
                                                                                   token_type_ids = segment_ids,
                                                                                   state_positions = state_position_ids,
                                                                                   attention_mask = input_mask,
                                                                                   slot_mask = slot_mask,
                                                                                   max_value = max_value,
                                                                                   op_ids = op_ids,
                                                                                   max_update = max_update)
                    if turn == 0:
                        sample_mask = None
                        loss_ans = loss_fnc(has_ans.view(-1, len(op2id)), op_ids.view(-1))
                        loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                                slot_ans_ids.contiguous(),
                                                                sample_mask = sample_mask,
                                                                slot_mask = None,
                                                                )
                        loss_s = masked_cross_entropy_for_value(start_logits.contiguous(),
                                                                start_idx.contiguous(),
                                                                sample_mask = sample_mask,
                                                                slot_mask = batch_cate_mask,
                                                                pad_idx = -1
                                                                )
                        loss_e = masked_cross_entropy_for_value(end_logits.contiguous(),
                                                                end_idx.contiguous(),
                                                                sample_mask = sample_mask,
                                                                slot_mask = batch_cate_mask,
                                                                pad_idx = -1)
                        loss = answer_weight * loss_ans + generation_weight * loss_g + extraction_weight * loss_s + extraction_weight * loss_e
                        weight_sum = answer_weight + (loss_g != 0).float() * 0.2 + (
                                loss_s !=  0).float() * extraction_weight + extraction_weight * (
                                             loss_e != 0).float()

                    elif turn == 1 or turn == 2:
                        if turn == 1:
                            sample_mask = (pred_ops.argmax(dim = -1) == 0)
                        else:
                            sample_mask = (op_ids == 0)
                        loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                                slot_ans_ids.contiguous(),
                                                                sample_mask = sample_mask
                                                                )
                        loss_s = masked_cross_entropy_for_value(start_logits.contiguous(),
                                                                start_idx.contiguous(),
                                                                sample_mask = sample_mask,
                                                                pad_idx = -1
                                                                )
                        loss_e = masked_cross_entropy_for_value(end_logits.contiguous(),
                                                                end_idx.contiguous(),
                                                                sample_mask = sample_mask,
                                                                pad_idx = -1)

                        loss = generation_weight * loss_g + extraction_weight * loss_s + extraction_weight * loss_e
                        weight_sum = 1

                    loss = loss / weight_sum if weight_sum != 0 else loss
                    batch_loss.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    enc_optimizer.step()
                    enc_scheduler.step()
                    model.zero_grad()
                    loss = loss.item()

                    if step % 100 == 0:
                        print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f," \
                              % (epoch + 1, args.n_epochs, step,
                                 len(train_dataloader), np.mean(batch_loss),
                                 loss))
                        batch_loss = []
                    if (step + 1) % 1000 == 0:
                        model.eval()
                        start_predictions = []
                        start_ids = []
                        end_predictions = []
                        end_ids = []
                        has_ans_predictions = []
                        has_ans_labels = []
                        gen_predictions = []
                        gen_labels = []
                        score_diffs = []
                        cate_score_diffs = []
                        score_noanses = []
                        all_input_ids = []
                        sample_ids = []
                        for step, batch in enumerate(dev_dataloader):
                            batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in
                                     batch]
                            input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, sid = batch
                            batch_size = input_ids.shape[0]
                            seq_lens = input_ids.shape[1]
                            if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                                teacher = gen_ids
                            else:
                                teacher = None

                            if turn == 0:
                                start_logits, end_logits, has_ans, gen_scores, _, _, _ = model(input_ids = input_ids,
                                                                                               token_type_ids = segment_ids,
                                                                                               state_positions = state_position_ids,
                                                                                               attention_mask = input_mask,
                                                                                               slot_mask = slot_mask,
                                                                                               max_value = max_value,
                                                                                               op_ids = op_ids,
                                                                                               max_update = max_update)
                                start_predictions +=  start_logits.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                end_predictions +=  end_logits.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                has_ans_predictions +=  has_ans.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                # has_ans_predictions+ = pred_ops.view(-1).cpu().detach().numpy().tolist()
                                gen_predictions +=  gen_scores.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                start_ids +=  start_idx.view(-1).cpu().detach().numpy().tolist()
                                end_ids +=  end_idx.view(-1).cpu().detach().numpy().tolist()
                                has_ans_labels +=  op_ids.view(-1).cpu().detach().numpy().tolist()
                                gen_labels +=  slot_ans_ids.view(-1).cpu().detach().numpy().tolist()
                                all_input_ids +=  input_ids.cpu().detach().numpy().tolist()

                            elif turn == 1:
                                start_logits, end_logits, has_ans, gen_scores, start_scores, end_scores, category_score = model(
                                    input_ids = input_ids,
                                    token_type_ids = segment_ids,
                                    state_positions = state_position_ids,
                                    attention_mask = input_mask,
                                    slot_mask = slot_mask,
                                    max_value = max_value,
                                    op_ids = op_ids,
                                    max_update = max_update)

                                start_predictions +=  start_logits.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                end_predictions +=  end_logits.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                # has_ans_predictions + =  has_ans.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                has_ans_predictions +=  pred_ops.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                gen_predictions +=  gen_scores.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                start_ids +=  start_idx.view(-1).cpu().detach().numpy().tolist()
                                end_ids +=  end_idx.view(-1).cpu().detach().numpy().tolist()
                                has_ans_labels +=  op_ids.view(-1).cpu().detach().numpy().tolist()
                                gen_labels +=  slot_ans_ids.view(-1).cpu().detach().numpy().tolist()
                                all_input_ids +=  input_ids.cpu().detach().numpy().tolist()
                                sample_ids +=  sid
                                score_d, cate_score, score_n = compute_jointscore(start_scores, end_scores, gen_scores, pred_ops, ans_vocab, slot_mask)
                                score_diffs += score_d
                                cate_score_diffs += cate_score
                                score_noanses += score_n


                            elif turn == 2:
                                start_logits, end_logits, has_ans, gen_scores, _, _, _ = model(input_ids = input_ids,
                                                                                               token_type_ids = segment_ids,
                                                                                               state_positions = state_position_ids,
                                                                                               attention_mask = input_mask,
                                                                                               slot_mask = slot_mask,
                                                                                               max_value = max_value,
                                                                                               op_ids = op_ids,
                                                                                               max_update = max_update)
                                start_predictions +=  start_logits.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                end_predictions +=  end_logits.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                has_ans_predictions +=  pred_ops.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                # has_ans_predictions+ = pred_ops.view(-1).cpu().detach().numpy().tolist()
                                gen_predictions +=  gen_scores.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
                                start_ids +=  start_idx.view(-1).cpu().detach().numpy().tolist()
                                end_ids +=  end_idx.view(-1).cpu().detach().numpy().tolist()
                                has_ans_labels +=  op_ids.view(-1).cpu().detach().numpy().tolist()
                                gen_labels +=  slot_ans_ids.view(-1).cpu().detach().numpy().tolist()
                                all_input_ids +=  input_ids[:, 1:].cpu().detach().numpy().tolist()
                                sample_ids +=  sid

                        if turn ==  0 or turn ==  2:
                            gen_acc, op_acc, opprec, op_recall, op_F1 = op_evaluation(start_predictions, end_predictions,
                                                                                       gen_predictions, has_ans_predictions,
                                                                                       start_ids, end_ids, gen_labels,
                                                                                       has_ans_labels, all_input_ids,
                                                                                       ans_vocab_nd, score_diffs = None,
                                                                                       cate_score_diffs = None,
                                                                                       score_noanses = None,
                                                                                       sketchy_weight = None,
                                                                                       verify_weight = None, sid = sample_ids,
                                                                                       catemask = cate_mask)
                        elif turn == 1:
                            gen_acc, op_acc, op_prec, op_recall, op_F1 = op_evaluation(start_predictions, end_predictions,
                                                                                       gen_predictions, has_ans_predictions,
                                                                                       start_ids, end_ids, gen_labels,
                                                                                       has_ans_labels, all_input_ids,
                                                                                       ans_vocab_nd, score_diffs,
                                                                                       cate_score_diffs, score_noanses,
                                                                                       sketchy_weight, verify_weight,
                                                                                       sample_ids, cate_mask)

                        file_logger.log(
                            "{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, loss,
                                                                                                                gen_acc,
                                                                                                                op_acc,
                                                                                                                op_prec,
                                                                                                                op_recall,
                                                                                                                op_F1,
                                                                                                                max(gen_acc,
                                                                                                                    best_score[
                                                                                                                        'gen_acc']),
                                                                                                                max(op_acc,
                                                                                                                    best_score[
                                                                                                                        'op_acc']),
                                                                                                                max(op_F1,
                                                                                                                    best_score[
                                                                                                                        'op_F1'])))
                        model_to_save = model.module if hasattr(model, 'module') else model
                        if turn == 1 or turn == 0:
                            saveOperationLogits(model, device=device, dataset=dev_dataloader, save_path=args.output_dir,
                                                turn=turn)
                            isbest = op_F1 > best_score['op_F1']
                        else:
                            isbest = gen_acc > best_score['gen_acc']
                        if isbest:
                            saved_name = 'model_best_turn' + str(turn) + '.bin'
                            best_score['op_acc'] = op_acc
                            best_score['gen_acc'] = gen_acc
                            best_score['op_F1'] = op_F1
                            save_path = os.path.join(args.save_dir, saved_name)
                            params = {
                                'model': model_to_save.state_dict(),
                                'optimizer': enc_optimizer.state_dict(),
                                'scheduler': enc_scheduler.state_dict(),
                                'args': args
                            }
                            torch.save(params, save_path)
                            file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}" \
                                            .format(epoch, gen_acc * 100, op_acc * 100, op_F1 * 100))
                        save_path = os.path.join(args.save_dir, 'checkpoint_epoch_' + str(epoch) + '.bin')
                        torch.save(model_to_save.state_dict(), save_path)
                        print(
                            "Best Score : generate_accurate : %.3f, operation_accurate : %.3f,operation_precision : %.3f, operation_recall:%.3f,operation_F1 : %.3f" % (
                                gen_acc, op_acc, op_prec, op_recall, op_F1))
                        print("\n")
                        model.train()
        else:
            scores= model_evaluation(model, dev_data_raw, tokenizer, slot_meta, 0, slot_ans=ontology, op_code = args.op_code, ans_vocab = ans_vocab_nd, cate_mask = cate_mask)

if __name__ ==  "__main__":
    main()
