import transformers
import torch
import numpy as np
from tqdm import tqdm
import pdb
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from argparse import ArgumentParser
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if np.array_equal(l[ind:ind+sll], sl): # check array is equal
            results.append((ind,ind+sll-1))

    return results

def load_data(args):
    index = []
    text = []
    cause = []
    effect = []

    with open(args.path, 'r') as file:
        train_data = csv.reader(file, delimiter=';')
        t = 0



        try:
            args.train
            for row in train_data:

                if t:
                    if row[0] == '0009.00005':
                        continue
                    index += [row[0]]
                    text += [row[1]]
                    cause += [row[2]]
                    effect += [row[3]]
                t += 1

        except:
            for row in train_data:

                if t:

                    index += [row[0]]
                    text += [row[1]]
                t += 1

    return index, text, cause, effect

def tokenization(text, tokenizer, args):
    context = []
    text_attention_mask = []

    for i in tqdm(range(len(text))):

        if args.roberta:
            text[i] = ' ' + text[i]

        token_text = tokenizer.tokenize(text[i])
        token_text_id = tokenizer.convert_tokens_to_ids(token_text)
        bert_input = tokenizer.prepare_for_model(ids=token_text_id,
                                                max_length=512,
                                                pad_to_max_length=True)
        
        input_ids = np.array(bert_input['input_ids'])
        attention_mask = np.array(bert_input['attention_mask'])
        context += [input_ids]
        text_attention_mask += [attention_mask]

    return context, text_attention_mask

def make_label(context, cause, effect, tokenizer, args): # context is token id

    print('make_label')

    cause_start = []
    cause_end = []
    effect_start = []
    effect_end = []

    for i in range(len(context)):

        if args.roberta:
            cause[i] = ' ' + cause[i]
            effect[i] = ' ' + effect[i]

        # tokenize cause and effect
        cause_token = tokenizer.tokenize(cause[i])
        cause_token_id = tokenizer.convert_tokens_to_ids(cause_token)
        
        cause_range = find_sub_list(cause_token_id, context[i])
        try:
            cause_start += [cause_range[0][0]]
            cause_end += [cause_range[0][1]]
        except: 
            print(cause[i])
            pdb.set_trace()

        effect_token = tokenizer.tokenize(effect[i])
        effect_token_id = tokenizer.convert_tokens_to_ids(effect_token)
        
        effect_range = find_sub_list(effect_token_id, context[i])
        try:
            effect_start += [effect_range[0][0]]
            effect_end += [effect_range[0][1]]
        except:
            print(effect[i])
            pdb.set_trace()
    
    cause_start = np.array(cause_start, dtype=np.float)
    cause_end = np.array(cause_end, dtype=np.float)
    effect_start = np.array(effect_start, dtype=np.float)
    effect_end = np.array(effect_end, dtype=np.float)

    return cause_start, cause_end, effect_start, effect_end

def save_preprocessing(context, text_attention_mask, cause_start, cause_end, effect_start, effect_end, args):
    
    if args.train:
        np.save('./data/text.npy', context)
        np.save('./data/text_attention_mask.npy', text_attention_mask)
        np.save('./data/cause_start.npy', cause_start)
        np.save('./data/cause_end.npy', cause_end)
        np.save('./data/effect_start.npy', effect_start)
        np.save('./data/effect_end.npy', effect_end)

    else:
        np.save('./data/test.npy', context)
        np.save('./data/test_attention_mask.npy', text_attention_mask)

def argparser():

    parser = ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--bert', action='store_true', default=False, help='whether using bert tokenizer for preprocessing')
    parser.add_argument('--roberta', action='store_true', default=False)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argparser()

    if args.roberta:
        model_version = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=True)

    elif args.bert:
        model_version = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    index, text, cause, effect = load_data(args)
    context, text_attention_mask = tokenization(text, tokenizer, args)
    cause_start, cause_end, effect_start, effect_end = make_label(context, cause, effect, tokenizer, args)
    save_preprocessing(context, text_attention_mask, cause_start, cause_end, effect_start, effect_end, args)
