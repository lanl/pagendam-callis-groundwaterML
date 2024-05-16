# TESTING FOR CSIRO GROUNDWATER CNNS
"""
"""

###############################################
## Packages
###############################################
import pytest
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('./'))
import model_definition as model

###############################################
## Define Fixtures
###############################################
@pytest.fixture
def kernel():
	return 3

@pytest.fixture
def local_neurons():
	return 8

@pytest.fixture
def local_depth():
	return 3

@pytest.fixture
def dense_neurons():
	return 32

@pytest.fixture
def dense_depth():
	return 4

@pytest.fixture
def bias():
	return True

@pytest.fixture
def batchsize():
	return 13

@pytest.fixture
def device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################
## Test Local Effects Module
###############################################
class Test_LocalEffects():
	""" """
	def test_shape_12chan(self, kernel, local_neurons, local_depth, bias, batchsize, device):
		""" Tests that the Local Effects Module outputs the correct shape for Precipitation and MODIS PET input shapes """
		in_chan = 12
		inpt = torch.rand(batchsize, in_chan, 9, 9)
		outpt_size = local_neurons * 9 * 9
		LE = model.Local_Effects(in_chan = in_chan,
								 kernel=kernel,
								 neurons = local_neurons,
								 bias = bias,
								 activation = nn.ReLU,
								 depth = local_depth)
		LE.to(device)
		outpt = LE(device, inpt)
		assert outpt.shape == torch.Size([batchsize, outpt_size])

	def test_shape_01chan(self, kernel, local_neurons, local_depth, bias, batchsize, device):
		""" Tests that the Local Effects Module outputs the correct shape for DEM input shape """
		in_chan = 1
		inpt = torch.rand(batchsize, in_chan, 9, 9)
		outpt_size = local_neurons * 9 * 9
		LE = model.Local_Effects(in_chan = in_chan,
								 kernel=kernel,
								 neurons = local_neurons,
								 bias = bias,
								 activation = nn.ReLU,
								 depth = local_depth)
		LE.to(device)
		outpt = LE(device, inpt)
		assert outpt.shape == torch.Size([batchsize, outpt_size])

	def test_invalid(self, kernel, local_neurons, bias, batchsize, device):
		""" Tests that the Local Effects Module errors out when the depth is less than 1 """
		with pytest.raises(AssertionError):
			in_chan = 12
			inpt = torch.rand(batchsize, in_chan, 9, 9)
			LE = model.Local_Effects(in_chan = in_chan,
									 kernel=kernel,
									 neurons = local_neurons,
									 bias = bias,
									 activation = nn.ReLU,
									 depth = 0)
			LE.to(device)
			outpt = LE(device, inpt)

###############################################
## Test Sub Architecture 1 (Image Processing)
###############################################
class Test_Sub1():
	""" """
	def test_shape(self, kernel, local_neurons, local_depth, dense_neurons, dense_depth, bias, batchsize, device):
		""" Tests that Sub Architecture 1 outputs the correct number of predictions (mean and std) """
		sample = {}
		sample['precip_scaled'] = torch.rand(batchsize, 12, 9, 9)
		sample['PET_scaled'] = torch.rand(batchsize, 12, 9, 9)
		sample['DEM_scaled'] = torch.rand(batchsize, 1, 9, 9)
		
		s1 = model.Sub1(local_effects_kernel = kernel,
						 local_effects_neurons = local_neurons,
						 local_effects_bias = bias,
						 local_effects_depth = local_depth,
						 activation = nn.ReLU,
						 dense_neurons = dense_neurons,
						 dense_bias = bias,
						 dense_depth = dense_depth)
		s1.to(device)
		outpt = s1(device, sample)
		assert outpt.shape == torch.Size([batchsize, 2])


	def test_invalid(self, kernel, local_neurons, local_depth, dense_neurons, bias, batchsize, device):
		""" Tests that Sub Architecture 1 errors out when the dense depth is less than 1 """
		with pytest.raises(AssertionError):
			sample = {}
			sample['precip_scaled'] = torch.rand(batchsize, 12, 9, 9)
			sample['PET_scaled'] = torch.rand(batchsize, 12, 9, 9)
			sample['DEM_scaled'] = torch.rand(batchsize, 1, 9, 9)
			
			s1 = model.Sub1(local_effects_kernel = kernel,
							 local_effects_neurons = local_neurons,
							 local_effects_bias = bias,
							 local_effects_depth = local_depth,
							 activation = nn.ReLU,
							 dense_neurons = dense_neurons,
							 dense_bias = bias,
							 dense_depth = 0)
			s1.to(device)
			outpt = s1(device, sample)

###############################################
## Test Sub Architecture 2 (Scalar Processing)
###############################################
class Test_Sub2():
	""" """
	def test_shape(self, dense_neurons, dense_depth, bias, batchsize, device):
		""" Tests that Sub Architecture 2 outputs the correct number of predictions (mean and std) """
		sample = {}
		sample['year_scaled'] = torch.rand(batchsize, 1)
		sample['coordinates_scaled'] = torch.rand(batchsize, 2)
		
		s2 = model.Sub2(activation = nn.ReLU,
						 dense_neurons = dense_neurons,
						 dense_bias = bias,
						 dense_depth = dense_depth)
		s2.to(device)
		outpt = s2(device, sample)
		assert outpt.shape == torch.Size([batchsize, 2])


	def test_invalid(self, dense_neurons, bias, batchsize, device):
		""" Tests that Sub Architecture 2 errors out when the dense depth is less than 1 """
		with pytest.raises(AssertionError):
			sample = {}
			sample['year_scaled'] = torch.rand(batchsize, 1)
			sample['coordinates_scaled'] = torch.rand(batchsize, 2)
			
			s2 = model.Sub2(activation = nn.ReLU,
							 dense_neurons = dense_neurons,
							 dense_bias = bias,
							 dense_depth = 0)
			s2.to(device)
			outpt = s2(device, sample)

###############################################
## Test CNN
###############################################
class Test_CNN():
	""" """
	def test_shape(self, kernel, local_neurons, local_depth, dense_neurons, dense_depth, bias, batchsize, device):
		""" Tests that CNN outputs the correct number of predictions (mean and std) """
		sample = {}
		sample['precip_scaled'] = torch.rand(batchsize, 12, 9, 9)
		sample['PET_scaled'] = torch.rand(batchsize, 12, 9, 9)
		sample['DEM_scaled'] = torch.rand(batchsize, 1, 9, 9)
		sample['year_scaled'] = torch.rand(batchsize, 1)
		sample['coordinates_scaled'] = torch.rand(batchsize, 2)
		
		CNN = model.CNN(local_effects_kernel = kernel,
						 local_effects_neurons = local_neurons,
						 local_effects_bias = bias,
						 local_effects_depth = local_depth,
						 activation = nn.ReLU,
						 dense_neurons = dense_neurons,
						 dense_bias = bias,
						 dense_depth = dense_depth)
		CNN.to(device)
		outpt = CNN(device, sample)
		assert outpt.shape == torch.Size([batchsize, 4])
