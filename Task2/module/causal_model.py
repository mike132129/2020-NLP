from transformers import BertTokenizer, BertModel
from transformers.modeling_utils import SequenceSummary
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
import pdb
import random

class causal_modified_xlnet(nn.Module):
    def __init__(self, model, pretrained_config):
        super(causal_modified_xlnet, self).__init__()
        self.cause_linear = nn.Linear(768, 2)
        self.effect_linear = nn.Linear(768, 2)
        self.model = model
        self.sequence_summary = SequenceSummary(pretrained_config)

    def forward(self, input_ids, attention_mask, labels):

        xlnet_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = xlnet_output[0]
        cause_logits = self.cause_linear(output)
        effect_logits = self.effect_linear(output)

        c_start_logits, c_end_logits = cause_logits.split(1, dim=-1)
        e_start_logits, e_end_logits = effect_logits.split(1, dim=-1)

        c_start_logits = c_start_logits.squeeze(-1)
        c_end_logits = c_end_logits.squeeze(-1)
        e_start_logits = e_start_logits.squeeze(-1)
        e_end_logits = e_end_logits.squeeze(-1)
        
        if labels == None:
            return c_start_logits, c_end_logits, e_start_logits, e_end_logits

        c_start_target = labels[0]
        c_end_target = labels[1]
        e_start_target = labels[2]
        e_end_target = labels[3]

        loss_fct = CrossEntropyLoss()
        c_start_loss = loss_fct(c_start_logits, c_start_target.long())
        c_end_loss = loss_fct(c_end_logits, c_end_target.long())
        e_start_loss = loss_fct(e_start_logits, e_start_target.long())
        e_end_loss = loss_fct(e_end_logits, e_end_target.long())
        
        return (c_start_loss, c_end_loss, e_start_loss, e_end_loss), (c_start_logits, c_end_logits, e_start_logits, e_end_logits)

class causal_modified_bert(nn.Module):
	def __init__(self, model):
		super(causal_modified_bert, self).__init__()

		self.model = model
		self.cause_linear = nn.Linear(768, 2)
		self.effect_linear = nn.Linear(768, 2)

	def forward(self, input_ids, attention_mask, labels):
		
		bert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

		sequence_output = bert_output[0]
		cause_logits = self.cause_linear(sequence_output)
		effect_logits = self.effect_linear(sequence_output)

		c_start_logits, c_end_logits = cause_logits.split(1, dim=-1)
		e_start_logits, e_end_logits = effect_logits.split(1, dim=-1)

		c_start_logits = c_start_logits.squeeze(-1)
		c_end_logits = c_end_logits.squeeze(-1)
		e_start_logits = e_start_logits.squeeze(-1)
		e_end_logits = e_end_logits.squeeze(-1)

		if labels == None: # predicting
			return c_start_logits, c_end_logits, e_start_logits, e_end_logits

		c_start_target = labels[0]
		c_end_target = labels[1]
		e_start_target = labels[2]
		e_end_target = labels[3]

		loss_fct = CrossEntropyLoss()
		c_start_loss = loss_fct(c_start_logits, c_start_target.long())
		c_end_loss = loss_fct(c_end_logits, c_end_target.long())
		e_start_loss = loss_fct(e_start_logits, e_start_target.long())
		e_end_loss = loss_fct(e_end_logits, e_end_target.long())

		return (c_start_loss, c_end_loss, e_start_loss, e_end_loss), (c_start_logits, c_end_logits, e_start_logits, e_end_logits)


