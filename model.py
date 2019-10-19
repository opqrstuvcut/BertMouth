#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
from transformers.modeling_bert import BertForTokenClassification


class BertMouth(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids, attention_mask):
        output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                              output_all_encoded_layers=False)

        output = self.dropout(output)
        logits = self.classifier(output)

        return logits

