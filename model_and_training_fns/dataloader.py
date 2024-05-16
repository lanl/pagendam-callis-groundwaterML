# PYTORCH DATALOADER FOR CSIRO GROUNDWATER DATA
"""
LANL / CSIRO Collaboration

Designed for the data accesible at https://doi.org/10.25919/skw8-yx65

"""
####################################
## Packages
####################################
import os
import sys
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

####################################
## DataSet Class
####################################
class GW_DataSet(Dataset):
	def __init__(self, data_dir: str):

		""" HELP

			Args:
				data_dir (str): path to directory containing all data

		"""

		## Model Arguments 
		self.data_dir = data_dir

		## Seperate Individual Files
		self.file_list = glob.glob(os.path.join(data_dir, '*'))
		self.file_keys = ['PET_scaled',
							'precip_scaled',
							'GW_depth',
							'year_scaled',
							'DEM_scaled',
							'coordinates_scaled']
		for key in self.file_keys:
			path = [f for f in self.file_list if key in f]
			err_msg = ''
			if len(path)<1: 
				err_msg = 'No files for '+key+' found in '+self.data_dir
			elif len(path)>1: 
				err_msg = 'Multiple files for '+key+' found in '+self.data_dir
			assert len(path)==1, err_msg
			setattr(self, key+'_path', path[0])
			setattr(self, key, np.load(getattr(self, key+'_path')))

		## Get Number of Samples
		self.Nsamples = self.GW_depth.shape[0]

	def __len__(self):
		"""
		Return number of samples in dataset.
		"""
		return self.Nsamples

	def __getitem__(self, index):
		"""
		Return a dictionary of all groundwater properties for a given index
		"""
		sample = {}
		for key in self.file_keys:
			try:
				sample[key] = torch.tensor(getattr(self, key)[index, :]).to(torch.float32)
			except: 
				sample[key] = torch.tensor(getattr(self, key)[index]).to(torch.float32)

		return sample

def make_dataloader(dataset: torch.utils.data.Dataset,
						batch_size: int=8,
						num_batches: int=100):
		""" Function to create a pytorch dataloader from a pytorch dataset
				https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
			Each dataloader has batch_size*num_batches samples randomly selected from the dataset

			num_workers: behavior training on dodona
				=0 if not specified, data is loaded in the main process;
					trains slower if multiple models being trained on the same node
				=1 seperates the data from the main process;
					training speed unaffected by multiple models being trained
				=2 splits data across 2 processors;
					cuts training time in half from num_workers=1
				>2 diminishing returns on training time

			persistant_workers:
				training time seems minimally affected, slight improvement when =True

			Args:
				dataset(torch.utils.data.Dataset): dataset to sample from for data loader
				batch_size (int): batch size
				num_batches (int): number of batches to include in data loader

			Returns:
				dataloader (torch.utils.data.DataLoader): pytorch dataloader made from calico model dataset
		"""
		randomsampler = RandomSampler(dataset, num_samples=batch_size*num_batches)
		dataloader = DataLoader(dataset, batch_size=batch_size, sampler=randomsampler, num_workers=4, persistent_workers=True)

		return dataloader