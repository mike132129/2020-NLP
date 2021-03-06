import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
from transformers import XLNetForSequenceClassification
from transformers import RobertaForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
import pdb
import tqdm
from argparse import ArgumentParser
from dataset import ques_ans_dataset, create_mini_batch
from makedata import *
import time
from module.cls_fin import modified_bert, modified_XLNet
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd



def argparser():

    parser = ArgumentParser()
    parser.add_argument('--test_data_path')
    parser.add_argument('--load_model')
    parser.add_argument('--output_path')
    parser.add_argument('--mode')
    parser.add_argument('--xlnet', action='store_true', default=False)
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--roberta', action='store_true', default=False)

    args = parser.parse_args()

    return args

def train(trainset, args, BATCH_SIZE=2):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)


    if args.xlnet:
        model_version = 'xlnet-base-cased'
        model = XLNetModel.from_pretrained(model_version)
        # pre_model = XLNetForSequenceClassification.from_pretrained(model_version, num_labels=2)

        cus_model = XLNetForSequenceClassification.from_pretrained(model_version, num_labels=2)
        # cus_model.config.summary_last_dropout = 0.3
        # cus_model = XLNetForSequenceClassification(cus_model.config)

    elif args.bert:
        model_version = 'bert-base-uncased'
        model = BertModel.from_pretrained(model_version)
        cus_model = modified_bert(model, device)

    elif args.roberta:
        model_version = 'roberta-base'
        model = RobertaModel.from_pretrained(model_version)
        cus_model = modified_bert(model, device)

    model.to(device)
    cus_model.to(device)

    train_set, valid_set = torch.utils.data.random_split(trainset, [9000, 1837])

    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    validloader = DataLoader(valid_set, batch_size=30, collate_fn=create_mini_batch)
    epochs = 30

    total_steps = len(dataloader) * epochs

    # optimizer = AdamW(cus_model.parameters(), lr=2e-5, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                         num_warmup_steps = 0, # Default value in run_glue.py
    #                                         num_training_steps = total_steps)

    param_optimizer = list(cus_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.0}
    # ]
    optimizer = AdamW(cus_model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = 0, # Default value in run_glue.py
                                        num_training_steps = total_steps)


    for epoch in range(epochs):
        print('==================Epoch{}==================='.format(epoch))
        model.train()
        cus_model.train()
        total_loss = 0

        ####### FREEZE LAYER
        if args.xlnet:
            pass
            # print('Freeze embedding and first layer')
            # for i, [j, k] in enumerate(cus_model.named_parameters()):
            #     if i < 34:
            #         k.requires_grad = False
        

        else:
            print('FREEZEEEEEEE')
            for i, [j, k] in enumerate(cus_model.named_parameters()):
                if i < 5:
                    print(j)
                    k.requires_grad = False
        ############ setting weight decay
        

        
        ############


        for step, data in enumerate(tqdm(dataloader)):

            if step % 1000 == 0 and not step == 0:
                print(' Loss: {}'.format(total_loss/step))

            tensors = [t.to(torch.device('cpu')) for t in data if t is not None]

            token_tensors, mask_tensors, labels = tensors[0], tensors[1], tensors[2]

            if args.xlnet:
                labels = torch.tensor(labels, dtype=torch.float32).to(device).long()
                mask_tensors = torch.tensor(mask_tensors, dtype=torch.long).to(device)
                token_tensors = token_tensors.to(device)
            else:
                labels = labels.float().to(device)
                mask_tensors = mask_tensors.to(device)
                token_tensors = token_tensors.to(device)


            loss, logits = cus_model(input_ids=token_tensors,
                                    attention_mask=mask_tensors,
                                    labels=labels
                                    )

            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cus_model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            cus_model.zero_grad()
            model.zero_grad()

        model.eval()
        cus_model.eval()

        with torch.no_grad():

            correct = 0

            val_loss = 0
            total = 0
            predict = []
            true = []

            for step, data in enumerate(tqdm(validloader)):

                tensors = [t.to(torch.device('cpu')) for t in data if t is not None]

                token_tensors, mask_tensors, labels = tensors[0], tensors[1], tensors[2]

                if args.xlnet:
                    labels = torch.tensor(labels, dtype=torch.float32).to(device).long()
                    mask_tensors = torch.tensor(mask_tensors, dtype=torch.long).to(device)
                    token_tensors = token_tensors.to(device)
                else:
                    labels = labels.float().to(device)
                    token_tensors = token_tensors.to(device)
                    mask_tensors = mask_tensors.to(device)

                loss, logits = cus_model(input_ids=token_tensors,
                                        attention_mask=mask_tensors,
                                        labels=labels
                                        )

                val_loss += loss.item()
                sigmoid = nn.Sigmoid()
                if args.xlnet:
                    prediction = torch.tensor([0 if logit[0] > logit[1] else 1 for logit in logits])
                else:
                    prediction = logits.sigmoid().round().cpu()
                correct += (prediction == labels.cpu().detach()).sum().item()
                total += prediction.size(0)
                predict += prediction.tolist()
                true += labels.cpu()

            
            precision, recall, f1, _ = precision_recall_fscore_support(true, predict, labels=[0, 1], average='weighted') 

            print('Validation Loss: {} at {} epoch'.format(val_loss/step, epoch))
            print('Accuracy: {}'.format(correct/total))
            print('F1 precision: {}, F1 score: {}'.format(precision, f1))

        torch.save(cus_model.state_dict(), './model/roberta-epoch-%s.pth' % epoch)


def predict(args, BATCH_SIZE):

    print('======================predict=======================')

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.xlnet:
        model_version = 'xlnet-base-cased'
        tokenizer = XLNetTokenizer.from_pretrained(model_version, do_lower_case=True)
        model = XLNetModel.from_pretrained(model_version)
        # cus_model = modified_XLNet(model, device, model.config)
        cus_model = XLNetForSequenceClassification.from_pretrained(model_version, num_labels=2)

    elif args.bert:
        model_version = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
        model = BertModel.from_pretrained(model_version)
        cus_model = modified_bert(model, device)

    elif args.roberta:
        model_version = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=True)
        model = RobertaModel.from_pretrained(model_version)
        cus_model = modified_bert(model, device)

    # Test data preprocess
    index, text, _ = load_data(args.test_data_path)
    context, attention_mask = tokenization(text, tokenizer, args)
    np.save('./data/test.npy', context)
    np.save('./data/test_mask.npy', attention_mask)

    model.to(device)
    cus_model.load_state_dict(torch.load(args.load_model, map_location='cuda:1'))
    cus_model.to(device)

    data_set = ques_ans_dataset(mode='test')
    dataloader = DataLoader(data_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    predict = []

    with torch.no_grad():

        for data in tqdm(dataloader):

            data = [t.to(device) for t in data if t is not None]

            tokens_tensors, masks_tensors = data
            logits = cus_model(input_ids=tokens_tensors,
                            attention_mask=masks_tensors,
                            labels=None
                            )
            sigmoid = nn.Sigmoid()
            if args.xlnet:
                prediction = torch.tensor([0 if logit[0] > logit[1] else 1 for logit in logits[0]])
            else:
                prediction = logits.sigmoid().round().cpu()
            predict += [int(i) for i in prediction.cpu().numpy().reshape(-1)]

    submission = pd.read_csv('./data/sample_submission.csv')
    submission['Index'][:] = index
    submission['Gold'][:] = predict
    submission.to_csv(args.output_path, index=False)


def main():

    args = argparser()

    if args.mode == 'train':

        trainset = ques_ans_dataset(mode=args.mode)
        train(trainset, args, BATCH_SIZE=4)

    else:

        predict(args, BATCH_SIZE=30)



        


if __name__ == '__main__':
    main()






