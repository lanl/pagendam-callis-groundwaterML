# TESTING FOR CSIRO GROUNDWATER DATASET / DATALOADER
"""
"""

###############################################
## Packages
###############################################
import pytest
import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('./'))
from model_definition import CNN
from dataloader import GW_DataSet, make_dataloader

###############################################
## Define Fixtures
###############################################
@pytest.fixture
def data_dir():
	return os.path.join('..', 'synthetic_data')

@pytest.fixture
def batch_size():
	return 5

@pytest.fixture
def num_batches():
	return 10

###############################################
## Tests GroundWater Dataset
###############################################
class Test_GWDataset():
	""" """
	def test_train(self, data_dir):
		""" Tests that dataset generates from training directory """
		subdir = os.path.join(data_dir, 'train')
		try:
			GW_DataSet(data_dir = subdir)
		except:
			assert False, "The GroundWater dataset raises error while in "+subdir
		else:
			assert True

	def test_tune(self, data_dir):
		""" Tests that dataset generates from tuning directory """
		subdir = os.path.join(data_dir, 'tune')
		try:
			GW_DataSet(data_dir = subdir)
		except:
			assert False, "The GroundWater dataset raises error while in "+subdir
		else:
			assert True

	def test_test(self, data_dir):
		""" Tests that dataset generates from testing directory """
		subdir = os.path.join(data_dir, 'test')
		try:
			GW_DataSet(data_dir = subdir)
		except:
			assert False, "The GroundWater dataset raises error while in "+subdir
		else:
			assert True

	def test_isdictionary(self, data_dir, batch_size, num_batches):
		""" Tests that dataloader returns an dictionary item """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert isinstance(sample, dict)

	def test_len(self, data_dir):
		""" Tests that dataset.__len__() operates as intended """
		subdir = os.path.join(data_dir, 'train')
		dataset = GW_DataSet(data_dir = subdir)
		assert dataset.__len__() == 100 #This number must match what is set in the generate_synth_data.ipynb notebook

	def test_getPET(self, data_dir):
		""" Tests that dataset.__getitem__() returns the correct size for PET """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert sample['PET_scaled'].shape == torch.Size([12, 9, 9])

	def test_getPrecip(self, data_dir):
		""" Tests that dataset.__getitem__() returns the correct size for precipitation """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert sample['precip_scaled'].shape == torch.Size([12, 9, 9])

	def test_getGWdepth(self, data_dir):
		""" Tests that dataset.__getitem__() returns the correct size for GroundWater depth """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert sample['GW_depth'].shape == torch.Size()

	def test_getYear(self, data_dir):
		""" Tests that dataset.__getitem__() returns the correct size for year """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert sample['year_scaled'].shape == torch.Size([1])

	def test_getDEM(self, data_dir):
		""" Tests that dataset.__getitem__() returns the correct size for DEM """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert sample['DEM_scaled'].shape == torch.Size([1, 9, 9])

	def test_getCoord(self, data_dir):
		""" Tests that dataset.__getitem__() returns the correct size for Coordinates """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		sample = dataset.__getitem__(1)
		assert sample['coordinates_scaled'].shape == torch.Size([2])


###############################################
## Tests Make DataLoader
###############################################
class Test_DataLoader():
	""" """
	def test_isdictionary(self, data_dir, batch_size, num_batches):
		""" Tests that dataloader returns an dictionary item """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		dataloader = make_dataloader(dataset,
									batch_size = batch_size,
									num_batches = num_batches)
		item = next(iter(dataloader))
		assert isinstance(item, dict)

	def test_batchsize(self, data_dir, batch_size, num_batches):
		""" Tests that dataloader returns an item with the correct batchsize """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		dataloader = make_dataloader(dataset,
									batch_size = batch_size,
									num_batches = num_batches)
		item = next(iter(dataloader))
		assert item['year_scaled'].shape == torch.Size([batch_size, 1])

	def test_numbatches(self, data_dir, batch_size, num_batches):
		""" Tests that dataloader returns the correct number of batches """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		dataloader = make_dataloader(dataset,
									batch_size = batch_size,
									num_batches = num_batches)
		assert len(dataloader) == num_batches

	def test_random_shuffle(self, data_dir, batch_size, num_batches):
		""" Tests that dataloader will shuffle data """
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		dataloader1 = make_dataloader(dataset,
									batch_size = 1,
									num_batches = num_batches)
		dataloader2 = make_dataloader(dataset,
									batch_size = 1,
									num_batches = num_batches)

		dataiter1 = iter(dataloader1)
		dataiter2 = iter(dataloader2)
		truth_array = []
		for b in range(num_batches):
			item1 = next(dataiter1)['coordinates_scaled'][0]
			item2 = next(dataiter2)['coordinates_scaled'][0]
			match = torch.equal(item1, item2)
			truth_array.append(match)

		assert not np.all(truth_array)

###############################################
## Tests Dataloader Interaction with Model
###############################################
class Test_DataLoader_in_Model():
	""" """
	def test_batching(self, data_dir):
		""" Tests that the batches are correctly read from the dataloader to the model """
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = CNN(local_effects_kernel=3,
					local_effects_neurons=8,
					local_effects_bias=True,
					local_effects_depth=3,
					activation=nn.ReLU,
					dense_neurons=32,
					dense_bias=True,
					dense_depth=4)

		batch_size = 3
		batches = 1
		subdir = os.path.join(data_dir, 'test')
		dataset = GW_DataSet(data_dir = subdir)
		dataloader = make_dataloader(dataset, batch_size, batches)

		for data in dataloader:
			model.eval()
			model.to(device)

			## Run as a batch
			GW_depth = data['GW_depth'].to(torch.float32).unsqueeze(-1).to(device)
			preds = model(device, data)

			## Split up into samples
			keys = data.keys()
			sample1 = {k: [] for k in keys}
			sample2 = {k: [] for k in keys}
			sample3 = {k: [] for k in keys}
			for k in keys:
				sample1[k] = data[k][0].unsqueeze(0)
				sample2[k] = data[k][1].unsqueeze(0)
				sample3[k] = data[k][2].unsqueeze(0)

			## Run on Each Sample
			preds1 = model(device, sample1)
			preds2 = model(device, sample2)
			preds3 = model(device, sample3)
			split_preds = torch.cat((preds1, preds2, preds3), dim=0)

		diff = (preds-split_preds).cpu().detach().numpy()
		diff = np.absolute(diff)

		assert np.all(diff < 1e-8)

