# MODEL DEFINITION FOR CSIRO GROUNDWATER CNNS
"""
LANL / CSIRO Collaboration

Model archiecture is based on the convolutional-based archiecture from https://doi.org/10.1016/j.spasta.2023.100740 

"""
###########################################
## Packages
###########################################
import torch
import torch.nn as nn

###########################################
## Local Effects Submodule
###########################################
class Local_Effects(nn.Module):
	def __init__(self,
				 in_chan: int=12,
				 kernel: int=3,
				 neurons: int=8,
				 bias: bool=True,
				 activation = nn.ReLU,
				 depth: int=3):
		
		""" Local Effects Module

			Args:
				in_chan (int): number of input channels;
							12 for Local Precipitation;
							12 for Local MODIS PET;
							1 for Local DEM
				kernel (int): size of 2D convoltional kernel
				neurons (int): number of neurons (equivalently "features" or "channels")
				bias (bool): determines if the 2D convolutions learn bias
				activation (nn.modules.activation): torch neural network layer class to use as activation
				depth (int): how many sets of Conv2D/Activation pairs before flattening
		"""

		super().__init__()

		## Define Paramaters
		self.in_chan = in_chan
		self.kernel = kernel
		self.neurons = neurons
		self.bias = bias 
		self.depth = depth
		assert self.depth >= 1, 'Depth must be equal to or greater than 1.'
		
		## Define Layers
		self.conv_layers = nn.ModuleList()
		self.conv_layers.append(nn.Conv2d(self.in_chan, self.neurons, self.kernel, bias=self.bias, padding='same'))
		for i in range(1, self.depth):
			self.conv_layers.append(nn.Conv2d(self.neurons, self.neurons, self.kernel, bias=self.bias, padding='same'))

		self.act_layers = nn.ModuleList()
		for i in range(self.depth):
			self.act_layers.append(activation())
	
	def forward(self, device, x):
		x = x.to(torch.float32).to(device)
		for i in range(self.depth):
			x = self.conv_layers[i](x)
			x = self.act_layers[i](x)

		x = torch.flatten(x, start_dim=1)
		return(x)

###########################################
## Sub Archiecture 1 (Image Processing)
###########################################
class Sub1(nn.Module):
	def __init__(self, 
				 local_effects_kernel: int=3,
				 local_effects_neurons: int=8,
				 local_effects_bias: bool=True,
				 local_effects_depth: int=3,
				 activation = nn.ReLU,
				 dense_neurons: int=32,
				 dense_bias: bool=True,
				 dense_depth: int = 4):

		""" Sub Archiecture 1 (Image Processing)

			Args:
				local_effects_kernel (int): size of 2D convoltional kernel for local effects modules
				local_effects_neurons (int): number of neurons (equivalently "features" or "channels") for local effects modules
				local_effects_bias (bool): determines if the 2D convolutions in the local effects modules learn bias
				local_effects_depth (int): how many sets of Conv2D/Activation pairs before flattening in the local effects modules
				activation (nn.modules.activation): torch neural network layer class to use as activation
				dense_neurons (int): number of neurons (equivalently "features" or "channels") in dense layers after local effects modules
				dense_bias (bool): determines if the dense layers learn bias
				dense_depth (int): how many dense layers (including the final prediction layer) there are after the local effects modules
		"""

		super().__init__()
	
		## Define Parameters
		self.dense_neurons = dense_neurons
		self.dense_bias = dense_bias
		self.dense_depth = dense_depth
		assert self.dense_depth >= 1, 'Dense Depth must be equal to or greater than 1.'
		self.concat_len = 9 * 9 * local_effects_neurons * 3 #9x9 image, times number of neurons, times 3 branches
		
		## Define Layers
		self.precipitation_branch = Local_Effects(in_chan = 12,
												 kernel = local_effects_kernel,
												 neurons = local_effects_neurons,
												 bias = local_effects_bias,
												 activation = activation,
												 depth = local_effects_depth)
		self.pet_branch = Local_Effects(in_chan = 12,
										 kernel = local_effects_kernel,
										 neurons = local_effects_neurons,
										 bias = local_effects_bias,
										 activation = activation,
										 depth = local_effects_depth)
		self.dem_branch = Local_Effects(in_chan = 1,
										 kernel = local_effects_kernel,
										 neurons = local_effects_neurons,
										 bias = local_effects_bias,
										 activation = activation,
										 depth = local_effects_depth)
		
		self.dense_layers = nn.ModuleList()
		self.dense_layers.append(nn.Linear(self.concat_len, self.dense_neurons, bias=self.dense_bias))
		for i in range(1, self.dense_depth-1):
			self.dense_layers.append(nn.Linear(self.dense_neurons, self.dense_neurons, bias=self.dense_bias))
		self.dense_layers.append(nn.Linear(self.dense_neurons, 2, bias=self.dense_bias))
		
		self.act_layers = nn.ModuleList()
		for i in range(self.dense_depth-1):
			self.act_layers.append(activation())

	def forward(self, device, sample):
		precipitation = self.precipitation_branch(device, sample['precip_scaled'])
		pet = self.pet_branch(device, sample['PET_scaled'])
		dem = self.dem_branch(device, sample['DEM_scaled'])
		x = torch.cat((precipitation, pet, dem), dim=1)

		for i in range(self.dense_depth-1):
			x = self.dense_layers[i](x)
			x = self.act_layers[i](x)
		x = self.dense_layers[i+1](x)
		return(x)

###########################################
## Sub Archiecture 2 (Scalar Processing)
###########################################
class Sub2(nn.Module):
	def __init__(self, 
				 activation = nn.ReLU,
				 dense_neurons: int=32,
				 dense_bias: bool=True,
				 dense_depth: int = 4):

		""" Sub Archiecture 2 (Scalar Processing)

			Args:
				activation(nn.modules.activation): torch neural network layer class to use as activation
				dense_neurons (int): number of neurons (equivalently "features" or "channels") in dense layers
				dense_bias (bool): determines if the dense layers learn bias
				dense_depth (int): how many dense layers (including the final prediction layer) are  included
		"""

		super().__init__()
	
		## Define Parameters
		self.dense_neurons = dense_neurons
		self.dense_bias = dense_bias
		self.dense_depth = dense_depth
		assert self.dense_depth >= 1, 'Dense Depth must be equal to or greater than 1.'

		## Define Layers
		self.dense_layers = nn.ModuleList()
		self.dense_layers.append(nn.Linear(3, self.dense_neurons, bias=self.dense_bias))
		for i in range(1, self.dense_depth-1):
			self.dense_layers.append(nn.Linear(self.dense_neurons, self.dense_neurons, bias=self.dense_bias))
		self.dense_layers.append(nn.Linear(self.dense_neurons, 2, bias=self.dense_bias))

		self.act_layers = nn.ModuleList()
		for i in range(self.dense_depth-1):
			self.act_layers.append(activation())

	def forward(self, device, sample):
		x = torch.cat((sample['year_scaled'], sample['coordinates_scaled']), dim=1).to(torch.float32).to(device)

		for i in range(self.dense_depth-1):
			x = self.dense_layers[i](x)
			x = self.act_layers[i](x)
		x = self.dense_layers[i+1](x)
		return(x)


###########################################
## Full CNN
###########################################
class CNN(nn.Module):
	def __init__(self, 
				 local_effects_kernel: int=3,
				 local_effects_neurons: int=8,
				 local_effects_bias: bool=True,
				 local_effects_depth: int=3,
				 activation = nn.ReLU,
				 dense_neurons: int=32,
				 dense_bias: bool=True,
				 dense_depth: int = 4):
		
		""" Full CNN

			Args:
				local_effects_kernel (int): size of 2D convoltional kernel for local effects modules
				local_effects_neurons (int): number of neurons (equivalently "features" or "channels") for local effects modules
				local_effects_bias (bool): determines if the 2D convolutions in the local effects modules learn bias
				local_effects_depth (int): how many sets of Conv2D/Activation pairs before flattening in the local effects modules
				activation (nn.modules.activation): torch neural network layer class to use as activation
				dense_neurons (int): number of neurons (equivalently "features" or "channels") in dense layers after local effects modules
				dense_bias (bool): determines if the dense layers learn bias
				dense_depth (int): how many dense layers (including the final prediction layer) there are after the local effects modules
		"""

		super().__init__()
		
		self.sub1 = Sub1(local_effects_kernel = local_effects_kernel,
						 local_effects_neurons = local_effects_neurons,
						 local_effects_bias = local_effects_bias,
						 local_effects_depth =  local_effects_depth,
						 activation = activation,
						 dense_neurons = dense_neurons,
						 dense_bias = dense_bias,
						 dense_depth = dense_depth)
		self.sub2 = Sub2(activation = activation,
						 dense_neurons = dense_neurons,
						 dense_bias = dense_bias,
						 dense_depth = dense_depth)

	def forward(self, device, sample):
		sub1 = self.sub1(device, sample)
		sub2 = self.sub2(device, sample)
		x = torch.cat((sub1, sub2), dim=1)
		return(x)