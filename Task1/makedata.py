import json
import numpy as np
import pandas as pd
import csv
import pdb
from transformers import BertTokenizer, XLNetTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import torch
# Load file

def load_data(path):
    index = []
    text = []
    gold = []

    with open(path, 'r') as file:
        train_data = csv.reader(file, delimiter=';')
        for row in train_data:
            index += [row[0]]
            text += [row[1]]
            try:
                gold += [int(row[2])]
            except:
                continue

    del index[0]; del text[0];

    return index, text, gold


def tokenization(text, tokenizer):

    print('tokenizing')

    context = []
    text_question_segment = []
    text_attention_mask = []


    for i in tqdm(range(len(text))):
        token_text = tokenizer.tokenize(text[i])
        token_text_id = tokenizer.convert_tokens_to_ids(token_text)
        bert_input = tokenizer.prepare_for_model(ids=token_text_id,
                                        max_length=512,
                                        pad_to_max_length=True
                                        )
    
        input_ids = np.array(bert_input['input_ids'])
        # token_types_ids = np.array(bert_input['token_types_ids'])
        attention_mask = np.array(bert_input['attention_mask'])

        context += [input_ids]
        # text_question_segment += [token_types_ids]
        text_attention_mask += [attention_mask]


    return context, text_attention_mask

def save_preprocessing(context, attention_mask, gold):
    np.save('./data/train.npy', context)
    np.save('./data/attention_mask.npy', attention_mask)
    np.save('./data/label.npy', np.array(gold))
    return




def argparser():

    parser = ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--xlnet', action='store_true', help='whether using xlnet tokenizer for preprocessing')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = argparser()

    index, text, gold = load_data(args.path)
    if args.xlnet:
        model_version = 'xlnet-base-cased'
        tokenizer = XLNetTokenizer.from_pretrained(model_version, do_lower_case=False)
    else:
        model_version = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    context, text_attention_mask = tokenization(text, tokenizer)
    save_preprocessing(context, text_attention_mask, gold)
