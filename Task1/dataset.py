from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import pdb

class ques_ans_dataset(Dataset):
    def __init__(self, mode):

        self.mode = mode

        assert self.mode in ['train', 'test']

        if self.mode ==  'train':   

            self.data = np.load('./data/train.npy', allow_pickle=True)
            self.mask = np.load('./data/attention_mask.npy', allow_pickle=True)
            self.label = np.load('./data/label.npy', allow_pickle=True)

        else:
            self.data = np.load('./data/test.npy', allow_pickle=True)
            self.mask = np.load('./data/test_mask.npy', allow_pickle=True)
        self.len = len(self.data)

    
    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode =='valid':

            ids = self.data[idx]
            label = self.label[idx]
            mask = self.mask[idx]

            tokens_tensor = torch.tensor(ids)
            label_tensor = torch.tensor(label)
            masks_tensors = torch.tensor(mask)

            return tokens_tensor, label_tensor, masks_tensors

        else:

            ids = self.data[idx]
            mask = self.mask[idx]

            label_tensor = None
            tokens_tensor = torch.tensor(ids)
            masks_tensors = torch.tensor(mask)

            return tokens_tensor, label_tensor, masks_tensors
        
    def __len__(self):
        return self.len
    

def create_mini_batch(samples):

    tokens_tensors = [s[0] for s in samples]
    masks_tensors = [s[2] for s in samples]

    if samples[0][1] is not None:
        label_ids = torch.stack([s[1] for s in samples])

    else:
        label_ids = None


    tokens_tensors = torch.stack(tokens_tensors)
    masks_tensors = torch.stack(masks_tensors)

    return tokens_tensors, masks_tensors, label_ids