#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
from transformers.modeling_bert import BertForTokenClassification


class BertMouth(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        output = output[0]

        output = self.dropout(output)
        logits = self.classifier(output)

        return logits

