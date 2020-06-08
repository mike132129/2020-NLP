from transformers import BertTokenizer, BertModel, XLNetModel
from transformers.modeling_utils import SequenceSummary
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import pdb

class modified_bert(nn.Module):
    def __init__(self, model, device):
        super(modified_bert, self).__init__()
        self.model = model
        self.cls_linear_1 = nn.Linear(768, 200)
        self.cls_linear_2 = nn.Linear(200, 1)
        self.device = device
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, labels):

        bert_output = self.model(input_ids=input_ids,attention_mask=attention_mask)
        pool_output = self.dropout_1(bert_output[1])
        cls_logits = self.cls_linear_1(pool_output)
        cls_logits = self.dropout_2(cls_logits)
        cls_logits = self.cls_linear_2(cls_logits)
        cls_logits = cls_logits.squeeze(-1)

        if labels == None: # when predicting
            return cls_logits

        pos_weight = torch.FloatTensor([10015/822]).to(self.device)
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight) # proportion of positive sample is 7

        cls_loss = criterion(cls_logits.view(-1, 1), labels.view(-1, 1))

        output = (cls_loss, cls_logits) 
        return output

class modified_XLNet(nn.Module):
    def __init__(self, model, device, pretrained_config):
        super(modified_XLNet, self).__init__()
        self.model = model
        self.cls_linear_1 = nn.Linear(768, 200)
        self.cls_linear_2 = nn.Linear(200, 1)
        self.device = device
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.sequence_summary = SequenceSummary(pretrained_config)

    def forward(self, input_ids, attention_mask, labels):

        xlnet_output = self.model(input_ids=input_ids,attention_mask=attention_mask)
        output = xlnet_output[0]
        output = self.sequence_summary(output)
        cls_logits = self.cls_linear_1(output)
        cls_logits = self.dropout_2(cls_logits)
        cls_logits = self.cls_linear_2(cls_logits)
        cls_logits = cls_logits.squeeze(-1)

        if labels == None: # when predicting
            return cls_logits

        pos_weight = torch.FloatTensor([10015/822]).to(self.device)
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight) # proportion of positive sample is 7

        cls_loss = criterion(cls_logits.view(-1, 1), labels.view(-1, 1))

        output = (cls_loss, cls_logits) 
        return output
