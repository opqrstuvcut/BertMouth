#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import logging

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
from tqdm import tqdm, trange
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule
from transformers import CONFIG_NAME, WEIGHTS_NAME
from torch import optim

from model import BertMouth
from data import make_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None,
                        type=str, help="file path for training.")
    parser.add_argument("--valid_file", default=None,
                        type=str, help="file path for validation.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization.")

    args = parser.parse_args()
    return args


def save(args, model, tokenizer, name):
    output_dir = os.path.join(args.output_dir, name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    tokenizer.save_vocabulary(output_dir)

    output_args_file = os.path.join(output_dir, 'training_args.bin')
    torch.save(args, output_args_file)


def train(args, tokenizer, device):
    logger.info("loading data")
    train_dataloader = make_dataloader(args.train_file, args.max_seq_length,
                                       args.train_batch_size, tokenizer)
    valid_dataloader = make_dataloader(args.valid_file, args.max_seq_length,
                                       args.train_batch_size, tokenizer)

    logger.info("building model")
    model = BertMouth.from_pretrained(args.bert_model,
                                      num_labels=len(tokenizer.vocab_size))
    model.to(device)

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    logger.info("setting optimizer")
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimization_steps = len(train_dataloader) * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=0,
                                     t_total=optimization_steps)
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    def calc_batch_loss(batch):
        batch = tuple(t.to(device) for t in batch)
        input_ids, y, input_mask, input_type_id = batch

        logits = model(input_ids, input_type_id, input_mask)
        logits = logits.view(-1, tokenizer.vocab_size)
        y = y.view(-1)
        loss = loss_fct(logits, y)
        return loss

    logger.info("train starts")
    loss_log_intervals = 5
    model.train()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        running_loss = 0.
        running_num = 0
        for step, batch in enumerate(train_dataloader):
            loss = calc_batch_loss(batch)
            loss.backward()

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_num += len(batch[0])
            if (step + 1) % loss_log_intervals == 0:
                print("[{0} epochs {1} / {2} batches] train loss: {3:.3g}".format(epoch + 1, step + 1,
                                                                                  len(train_dataloader),
                                                                                  running_loss / running_num))
                running_loss = 0.
                running_num = 0

        model.eval()
        valid_loss = 0.
        valid_num = 0
        for batch in valid_dataloader:
            valid_loss += calc_batch_loss(batch).item()
            valid_num += len(batch[0])
        print("[{0} epoch] valid loss: {1:.3g}".format(epoch + 1,
                                                       valid_loss / valid_num))

        model.train()

    save(args, model, tokenizer, "model")


def predict(args, tokenizer, device):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model_state_dict = torch.load(os.path.join(args.bert_model, "pytorch_model.bin"),
                                  map_location=device)
    model = BertMouth.from_pretrained(args.bert_model,
                                      state_dict=model_state_dict,
                                      num_labels=tokenizer.vocab_size)
    model.to(device)


def main():
    args = parse_argument()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if device != "cpu":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False,
                                              tokenize_chinese_chars=False)

    if args.do_train:
        train(args, tokenizer, device)
    if args.do_predict:
        predict(args, tokenizer, device)


if __name__ == '__main__':
    main()
