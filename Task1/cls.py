import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, XLNetModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
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
torch.manual_seed(1320)


def argparser():

    parser = ArgumentParser()
    parser.add_argument('--test_data_path')
    parser.add_argument('--load_model')
    parser.add_argument('--output_path')
    parser.add_argument('--mode')
    parser.add_argument('--xlnet', action='store_true')

    args = parser.parse_args()

    return args

def train(trainset, BATCH_SIZE=2):

    model_version = 'xlnet-base-cased' # or bert-base-uncased
    model = XLNetModel.from_pretrained(model_version) # or BertModel

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu') use cpu only
    print("device:", device)

    model.to(device)

    cus_model = modified_XLNet(model, device) # or modified_bert
    cus_model.to(device)

    train_set, valid_set = torch.utils.data.random_split(trainset, [10000, 837])


    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    validloader = DataLoader(valid_set, batch_size=30, collate_fn=create_mini_batch)
    epochs = 30

    total_steps = len(dataloader) * epochs

    optimizer = AdamW(cus_model.parameters(), lr=4e-6, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    for epoch in range(epochs):
        print('==================Epoch{}==================='.format(epoch))
        model.train()
        cus_model.train()
        total_loss = 0

        ####### FREEZE LAYER
        # print('FREEZEEEEEEE')
        for i, [j, k] in enumerate(cus_model.named_parameters()):
            if i < 21:
                k.requires_grad = False

        ###########

        for step, data in enumerate(tqdm(dataloader)):

            if step % 1000 == 0 and not step == 0:
                print(' Loss: {}'.format(total_loss/step))

            tensors = [t.to(device) for t in data if t is not None]

            token_tensors, mask_tensors, labels = tensors[0], tensors[1], tensors[2]
            loss, logits = cus_model(input_ids=token_tensors,
                                    attention_mask=mask_tensors,
                                    labels=labels.float()
                                    )

            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

                tensors = [t.to(device) for t in data if t is not None]

                token_tensors, mask_tensors, labels = tensors[0], tensors[1], tensors[2]

                loss, logits = cus_model(input_ids=token_tensors,
                                        attention_mask=mask_tensors,
                                        labels=labels.float()
                                        )

                val_loss += loss.item()
                sigmoid = nn.Sigmoid()
                prediction = sigmoid(logits).round()
                correct += (prediction == labels).sum().item()
                total += prediction.size(0)
                predict += prediction.cpu()
                true += labels.cpu()
            
            precision, recall, f1, _ = precision_recall_fscore_support(true, predict, labels=[0, 1], average='weighted') 

            print('Validation Loss: {} at {} epoch'.format(val_loss/step, epoch))
            print('Accuracy: {}'.format(correct/total))
            print('F1 precision: {}, F1 score: {}'.format(precision, f1))

        torch.save(cus_model.state_dict(), './model/xlnet-freeze-epoch-%s.pth' % epoch)


def predict(test_data_path, model_path, BATCH_SIZE):

    print('======================predict=======================')
    model_version = 'xlnet-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    index, text, _ = load_data(test_data_path)
    context, attention_mask = tokenization(text, tokenizer)
    np.save('./data/test.npy', context)
    np.save('./data/test_mask.npy', attention_mask)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = BertModel.from_pretrained(model_version)
    model.to(device)

    cus_model = modified_bert(model, device)
    cus_model.load_state_dict(torch.load(model_path))
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
            prediction = sigmoid(logits).round()
            predict += [int(prediction.cpu())]

    submission = pd.read_csv('./data/sample_submission.csv')
    submission['Index'][:] = index
    submission['Gold'][:] = predict
    submission.to_csv(r'result.csv', index=False)


def main():

    args = argparser()


    if args.mode == 'train':

        trainset = ques_ans_dataset(mode=args.mode)

        train(trainset, BATCH_SIZE=2)

    else:

        predict(test_data_path=args.test_data_path, model_path=args.load_model, BATCH_SIZE=1)



        


if __name__ == '__main__':
    main()






