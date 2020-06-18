import numpy
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, XLNetModel, RobertaModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn

from argparse import ArgumentParser
from dataset import causal_dataset, create_mini_batch
from module.causal_model import causal_modified_bert, causal_modified_xlnet
import pdb
from tqdm import tqdm
from makedata import *
import pandas


torch.manual_seed(1320)

def argparser():

    parser = ArgumentParser()
    parser.add_argument('--test_data_path')
    parser.add_argument('--load_model')
    parser.add_argument('--output_path')
    parser.add_argument('--mode')
    parser.add_argument('--roberta', default=False, action='store_true')
    parser.add_argument('--bert', default=False, action='store_true')
    parser.add_argument('--batch_size', default=3)
    parser.add_argument('--path')

    args = parser.parse_args()

    return args

def preprocess_test_data(args):

    if args.roberta:
        model_version = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=False)
    else:
        model_version = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    index, text, _, _ = load_data(args)
    context, text_attention_mask = tokenization(text, tokenizer, args)
    np.save('./data/test.npy', context)
    np.save('./data/test_attention_mask.npy', text_attention_mask)
    return index, text


def train(trainset, args, BATCH_SIZE):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # device = torch.device('cpu')
    
    if args.roberta:
        model_version = 'roberta-base'
        model = RobertaModel.from_pretrained(model_version)
        cus_model = causal_modified_bert(model)
    elif args.bert:
        model_version = 'bert-base-uncased' # or xlnet-base-cased
        model = BertModel.from_pretrained(model_version)
        cus_model = causal_modified_bert(model)

    model.to(device)
    cus_model.to(device)

    train_set, valid_set = random_split(trainset, [750, 50])

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    validloader = DataLoader(valid_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    epochs = 30

    total_steps = len(trainloader) * epochs

    optimizer = AdamW(cus_model.parameters(), lr=5e-6, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    for epoch in range(epochs):
        print('==================Epoch{}==================='.format(epoch))
        model.train()
        cus_model.train()
        total_c_start_loss = 0.0
        total_c_end_loss = 0.0
        total_e_start_loss = 0.0
        total_e_end_loss = 0.0

        total_loss = [total_c_start_loss, total_c_end_loss, total_e_start_loss, total_e_end_loss]

        ####### FREEZE LAYER
        # print('FREEZEEEEEEE')
        if args.roberta:
            pass
        elif args.bert:
            for i, [j, k] in enumerate(cus_model.named_parameters()):
                if i < 21:
                    k.requires_grad = False

        ###########

        for step, data in enumerate(tqdm(trainloader)):

            if step % 100 == 0 and not step == 0:
                print('  Total cause_Start Loss: {}, cause_End Loss: {}, effect_Start Loss: {}, effect_end Loss: {}'.\
                    format(total_loss[0]/len(trainloader), total_loss[1]/len(trainloader), total_loss[2]/len(trainloader), total_loss[3]/len(trainloader)))

            tensors = [t.to(device) for t in data if t is not None]

            token_tensors, masks_tensors, caus_labels = tensors
            
            loss, _ = cus_model(input_ids=token_tensors,
                                attention_mask=masks_tensors,
                                labels=caus_labels)

            for i in range(len(loss)):
                total_loss[i] += loss[i].item()

            loss = (loss[0] + loss[1] + loss[2])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cus_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            
            # Update learning rate schedule
            scheduler.step()

            cus_model.zero_grad()
            model.zero_grad()

        print('Validation...')

        val_c_start_loss = 0.0
        val_c_end_loss = 0.0
        val_e_start_loss = 0.0
        val_e_end_loss = 0.0

        val_loss = [val_c_start_loss, val_c_end_loss, val_e_start_loss, val_e_end_loss]
        model.eval()
        cus_model.eval()

        with torch.no_grad():

            for data in tqdm(validloader):

                tensors = [t.to(device) for t in data if t is not None]
                token_tensors, masks_tensors, caus_labels = tensors

                loss, _ = cus_model(input_ids=token_tensors,
                        attention_mask=masks_tensors,
                        labels=caus_labels)

                for i in range(len(loss)):
                    val_loss[i] += loss[i].item()

            print('  Total cause_Start Loss: {}, cause_End Loss: {}, effect_Start Loss: {}, effect_end Loss: {}'.\
                format(val_loss[0]/len(validloader), val_loss[1]/len(validloader), val_loss[2]/len(validloader), val_loss[3]/len(validloader)))

        torch.save(cus_model.state_dict(), './model/xlnet-freeze-epoch-lr-5e-7-%s.pth' % epoch)

def predict(index, text, args):

    dataset = causal_dataset(mode=args.mode)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.roberta:
        model_version = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=True)
        model = RobertaModel.from_pretrained(model_version)
        cus_model = causal_modified_bert(model)

    elif args.bert:

        model_version = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
        model = BertModel.from_pretrained(model_version)
        cus_model = causal_modified_bert(model)

    model.to(device)
    cus_model.load_state_dict(torch.load(args.load_model))
    cus_model.to(device)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_mini_batch)

    model.eval()
    cus_model.eval()

    cause = []
    effect = []

    with torch.no_grad():

        for data in tqdm(dataloader):

            tensors = [t.to(device) for t in data if t is not None]
            token_tensors, masks_tensors = tensors
            caus_labels = None

            logits = cus_model(input_ids=token_tensors,
                        attention_mask=masks_tensors,
                        labels=caus_labels)

            # context total length
            total_length = (masks_tensors != 0).sum().item()

            c_start, c_end, e_start, e_end = logits
            softmax = nn.Softmax(dim=-1)
            
            c_start_pred = softmax(c_start).argmax(dim=-1).item()
            c_end_pred = c_start_pred + softmax(c_end[0][c_start_pred : total_length]).argmax(dim=-1).item()

            e_start_pred = softmax(e_start).argmax(dim=-1).item()
            e_end_pred = e_start_pred + softmax(e_end[0][e_start_pred : total_length]).argmax(dim=-1).item()

            cause_pred = tokenizer.convert_ids_to_tokens(token_tensors[0][c_start_pred:c_end_pred], skip_special_tokens=True)
            effect_pred = tokenizer.convert_ids_to_tokens(token_tensors[0][e_start_pred:e_end_pred], skip_special_tokens=True)

            cause += [tokenizer.convert_tokens_to_string(cause_pred)]
            effect += [tokenizer.convert_tokens_to_string(effect_pred)]

    with open(args.output_path, 'w') as file:

        file.write('Index;Text;Cause;Effect\n')

        for i in range(len(text)):
            file.write(index[i] + ';' + text[i] + ';' + cause[i] + ';' + effect[i] + '\n')

        file.close()


def main():

    args = argparser()

    if args.mode == 'train':
        trainset = causal_dataset(mode=args.mode)
        train(trainset, args, BATCH_SIZE=int(args.batch_size))

    if args.mode == 'test':
        index, text = preprocess_test_data(args)
        predict(index, text, args)



if __name__ == '__main__':
    main()
