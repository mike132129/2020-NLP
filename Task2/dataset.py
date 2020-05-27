from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import pdb

class causal_dataset(Dataset):
	def __init__(self, mode):

		self.mode = mode

		if self.mode == 'train':

			self.data = np.load('./data/text.npy', allow_pickle=True)
			self.mask = np.load('./data/text_attention_mask.npy', allow_pickle=True)
			self.cause_start = np.load('./data/cause_start.npy', allow_pickle=True)
			self.cause_end = np.load('./data/cause_end.npy', allow_pickle=True)
			self.effect_start = np.load('./data/effect_start.npy', allow_pickle=True)
			self.effect_end = np.load('./data/effect_end.npy', allow_pickle=True)

		if self.mode == 'test':

			self.data = np.load('./data/test.npy', allow_pickle=True)
			self.mask = np.load('./data/test_attention_mask.npy', allow_pickle=True)

		self.len = len(self.data)

	def __getitem__(self, idx):

		if self.mode == 'train':

			ids = self.data[idx]
			mask = self.mask[idx]

			c_start = self.cause_start[idx]
			c_end = self.cause_end[idx]
			e_start = self.effect_start[idx]
			e_end = self.effect_end[idx]

			token_tensor = torch.tensor(ids)
			masks_tensor = torch.tensor(mask)
			c_start_tensor = torch.tensor(c_start)
			c_end_tensor = torch.tensor(c_end)
			e_start_tensor = torch.tensor(e_start)
			e_end_tensor = torch.tensor(e_end)

			return token_tensor, masks_tensor, c_start_tensor, c_end_tensor, e_start_tensor, e_end_tensor

		else:

			ids = self.data[idx]
			mask = self.mask[idx]

			token_tensor = torch.tensor(ids)
			masks_tensor = torch.tensor(mask)

			return token_tensor, masks_tensor

	def __len__(self):
		return self.len


def create_mini_batch(samples):
	token_tensor = [s[0] for s in samples]
	masks_tensor = [s[1] for s in samples]

	token_tensor = torch.stack(token_tensor)
	masks_tensor = torch.stack(masks_tensor)
	
	try:
		c_start_tensor = [s[2] for s in samples]
		c_end_tensor = [s[3] for s in samples]
		e_start_tensor = [s[4] for s in samples]
		e_end_tensor = [s[5] for s in samples]
		labels = torch.stack((torch.tensor(c_start_tensor), torch.tensor(c_end_tensor), torch.tensor(e_start_tensor), torch.tensor(e_end_tensor)), dim=0)

	except:
		labels = None

	return token_tensor, masks_tensor, labels
