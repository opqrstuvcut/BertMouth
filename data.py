#! /usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, unique_id, text):
        self.unique_id = unique_id
        self.text = text


class InputFeatures(object):
    def __init__(self, unique_id, tokens, input_type_ids, input_ids, input_mask):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_type_ids = input_type_ids
        self.input_ids = input_ids
        self.input_mask = input_mask


class BertMouthDataset(torch.utils.data.Dataset):
    def __init__(self, all_input_ids, all_input_mask, all_input_type_ids, all_num_tokens,
                 mask_id=4, ignore_index=0):
        self._data_num = len(all_input_ids)
        self._all_input_ids = all_input_ids
        self._all_input_mask = all_input_mask
        self._all_input_type_ids = all_input_type_ids
        self._all_num_tokens = all_num_tokens
        self._mask_id = mask_id
        self._ignore_index = ignore_index

    def __len__(self):
        return self._data_num

    def __getitem__(self, idx):
        input_ids = self._all_input_ids[idx].clone()
        input_mask = self._all_input_mask[idx]
        input_type_ids = self._all_input_type_ids[idx]
        num_tokens = self._all_num_tokens[idx]

        target_token_pos = np.random.randint(1, num_tokens - 1)

        y = [self._ignore_index] * len(input_ids)
        y[target_token_pos] = input_ids[target_token_pos]
        y = torch.tensor(y, dtype=torch.long)
        input_ids[target_token_pos] = self._mask_id

        return input_ids, y, input_mask, input_type_ids, target_token_pos


def read_texts(input_file):
    examples = []
    with open(input_file, "r", encoding='utf-8') as f:
        unique_id = 0
        for row in f:
            tokens = row.strip()

            examples.append(InputExample(unique_id=unique_id, text=tokens))
            unique_id += 1

    return examples


def convert_examples_to_features(examples, seq_length, tokenizer, ignore_index=0):
    def make_feature(ex_index, tokens):
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")

        input_type_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return InputFeatures(unique_id=ex_index,
                             tokens=tokens,
                             input_ids=input_ids,
                             input_type_ids=input_type_ids,
                             input_mask=input_mask)

    features = []
    for example in examples:
        subwords = tokenizer.tokenize(example.text)
        if len(subwords) > seq_length - 2:
            continue
        features.append(make_feature(example.unique_id, subwords))

    return features


def make_dataloader(file_path, max_seq_length, batch_size, tokenizer, eval_mode=False):
    examples = read_texts(file_path)
    features = convert_examples_to_features(examples,
                                            max_seq_length,
                                            tokenizer,
                                            ignore_index=0)

    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_input_type_ids = torch.tensor([f.input_type_ids for f in features],
                                      dtype=torch.long)
    all_num_tokens = [len(f.tokens) for f in features]

    mask_id = tokenizer.vocab["[MASK]"]
    data = BertMouthDataset(all_input_ids=all_input_ids,
                            all_input_mask=all_input_mask,
                            all_input_type_ids=all_input_type_ids,
                            all_num_tokens=all_num_tokens,
                            mask_id=mask_id,
                            ignore_index=0)

    for ex_index in [0, 1]:
        inputs_ids, y, input_mask, input_type_ids, target_token_pos = data[ex_index]
        ex_tokens = features[ex_index].tokens.copy()
        ex_tokens[target_token_pos] = "[MASK]"
        logger.info("*** Loaded Example ***")
        logger.info("guid: %s" % (ex_index))
        logger.info("masked tokens: %s" %
                    " ".join([str(x) for x in ex_tokens]))
        logger.info("input_ids: %s" % list(inputs_ids.numpy()))
        logger.info("y: %s" % list(y.numpy()))
        logger.info("input_mask: %s" % list(input_mask.numpy()))
        logger.info("segment_ids: %s" % list(input_type_ids.numpy()))

    if eval_mode:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


if __name__ == '__main__':
    pass
