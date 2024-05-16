# PYTORCH MODEL TRAINING FUNCTIONS
"""
Contains functions for training, validating, and testing a pytorch model.
"""
####################################
## Packages
####################################
import os
import sys
import glob
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

####################################
## Saving Helpful Functions
####################################
def save_append_df(path: str, df: pd.DataFrame, START: bool):
	""" Function to save/append dataframe contents to a csv file

		Args:
			path (str): path of csv file
			df (pd.DataFrame): pandas dataframe to save
			START (bool): indicates if the file path needs to be initiated

		Returns:
			No Return Objects
	"""
	if START:
		assert not os.path.isfile(path), 'If starting training, '+path+' should not exist.'
		df.to_csv(path, header=True, index = True, mode='x')
	else:
		assert os.path.isfile(path), 'If continuing training, '+path+' should exist.'
		df.to_csv(path, header=False, index = True, mode='a')

def append_to_dict(dictt: dict, batch_ID: int, truth, pred, loss):
	""" Function to appending sample information to a dictionary
		Dictionary must be initialized with correct keys

		Args:
			dictt (dict): dictionary to append sample information to
			batch_ID (int): batch ID number for samples
			truth (torch.tensor([[float], ...])): 
				shape: (batchsize, 1)
				array of GW_depth (ground truth) for batch of samples
			pred (torch.tensor([[float, float, float, float], ...])): 
				shape: (batchsize, 4)
				array of prediction values (mu1, log_sigma1, mu2, log_sigma2) for batch of samples
			loss (torch.tensor([[float], ...])): 
				shape: (batchsize, 1)
				array of loss values for batch of samples

		Returns:
			dictt (dict): dictionary with appended sample information
	"""
	batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]
	for i in range(batchsize):
		dictt["epoch"].append(0) # To be easily identified later
		dictt["batch"].append(batch_ID)
		dictt["truth"].append(truth.cpu().detach().numpy().flatten()[i])
		mu1, log_sigma1, mu2, log_sigma2 = pred[i]
		dictt["mu1_prediction"].append(mu1.cpu().detach().numpy().flatten()[0])
		dictt["mu2_prediction"].append(mu2.cpu().detach().numpy().flatten()[0])
		dictt["log_sigma1_prediction"].append(log_sigma1.cpu().detach().numpy().flatten()[0])
		dictt["log_sigma2_prediction"].append(log_sigma2.cpu().detach().numpy().flatten()[0])
		dictt["loss"].append(loss[i].cpu().detach().numpy().flatten()[0])

	return dictt

####################################
## Custom Loss Function
####################################
class Distribution_Loss(nn.Module):
	"""
	This is the loss function implemented as described in the paper orignally
	We suspect that it is numerically unstable
	"""
	def __init__(self):
		super(Distribution_Loss, self).__init__()

	def forward(self, preds: tuple[float, float, float, float], truth: float):
		"""
		Custom loss for groundwater data
		Args:
			pred (torch.tensor([[float, float, float, float], ...])): 
				shape: (batchsize, 4)
				array of prediction values (mu1, log_sigma1, mu2, log_sigma2) for batch of samples
			truth (torch.tensor([[float], ...])): 
				shape: (batchsize, 1)
				array of GW_depth (ground truth) for batch of samples
		Returns:
			loss (torch.tensor([[float], ...])): 
				shape: (batchsize, 1)
				array of loss values for batch of samples
		"""
		batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]
		loss = torch.zeros(batchsize)
		for i in range(batchsize):
			mu1, log_sigma1, mu2, log_sigma2 = preds[i]
			y = truth[i]
			sigma1 = torch.exp(log_sigma1)
			sigma2 = torch.exp(log_sigma2)
			mu = mu1 + mu2
			sigma_sqr = (sigma1**2) + (sigma2**2)
			f = ((2*math.pi)**(-1/2)) * ((torch.sqrt(sigma_sqr) * y)**(-1)) * torch.exp((-1) * ((torch.log(y) - mu)**2) * ((2 * sigma_sqr)**(-1)))
			loss[i] = torch.log(f)

		loss *= -1	
		return loss

class Distribution_Loss_v2(nn.Module):
	"""
	This is the loss function implemented as described in communications with Pagendam
	We hope that is is more stable and allows the models to train
	"""
	def __init__(self):
		super(Distribution_Loss_v2, self).__init__()

	def forward(self, preds: tuple[float, float, float, float], truth: float):
		"""
		Custom loss for groundwater data
		Args:
			pred (torch.tensor([[float, float, float, float], ...])): 
				shape: (batchsize, 4)
				array of prediction values (mu1, log_sigma1, mu2, log_sigma2) for batch of samples
			truth (torch.tensor([[float], ...])): 
				shape: (batchsize, 1)
				array of GW_depth (ground truth) for batch of samples
		Returns:
			loss (torch.tensor([[float], ...])): 
				shape: (batchsize, 1)
				array of loss values for batch of samples
		"""
		batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]
		loss = torch.zeros(batchsize)
		for i in range(batchsize):
			mu1, log_sigma1, mu2, log_sigma2 = preds[i]
			y = truth[i]
			sigma1 = torch.exp(log_sigma1)
			sigma2 = torch.exp(log_sigma2)
			mu = mu1 + mu2
			sigma = torch.sqrt((sigma1**2) + (sigma2**2))
			loss[i] = -0.5*math.log(2*math.pi) - torch.log(sigma) - torch.log(y) -0.5*(( (mu - torch.log(y)) / sigma )**2)
			loss[i] *= -1

		return loss

####################################
## Training on a Datastep
####################################
def train_datastep(data: dict, 
					model,
					optimizer,
					loss_fn, 
					device: torch.device):
	""" Function to complete a training step on a batch of samples

		Args:
			data (dict): dictionary of groundwater information
			model (loaded pytorch model): model to train
			optimizer (torch.optim): optimizer for training set
			loss_fn (torch.nn Loss Function): loss function
											  Distribution_Loss for the groundwater data
			device (torch.device): device index to select

		Returns:
			loss (): evaluated loss for the data sample
	"""
	## Set model to train
	model.to(device)
	model.train()

	## Extract data
	truth = data['GW_depth'].to(torch.float32).unsqueeze(-1).to(device)
	
	## Perform a forward pass
	preds = model(device, data).to(device)
	assert not torch.any(torch.isnan(preds)), 'Model returned "NaN" as prediction'
	loss = loss_fn(preds, truth)
	assert not torch.any(torch.isnan(loss)), 'Model made numerical prediction, but loss was calculated as "NaN"'

	## Perform backpropagation and update the weights
	optimizer.zero_grad()
	loss.sum().backward()
	optimizer.step()

	return truth, preds, loss

####################################
## Evaluating on a Datastep
####################################
def eval_datastep(data: tuple, 
					model,
					loss_fn,
					device: torch.device,):
	""" Function to complete a validation step on a batch of samples

		Args:
			data (tuple): tuple of model input and corresponding ground truth
			model (loaded pytorch model): model evaluate
			loss_fn (torch.nn Loss Function): loss function
											  Distribution_Loss for the groundwater data
			device (torch.device): device index to select

		Returns:
			loss (): evaluated loss for the data sample
	"""
	## Set model to eval
	model.to(device)
	model.eval()
	
	## Extract data
	truth = data['GW_depth'].to(torch.float32).unsqueeze(-1).to(device)
	
	## Perform a forward pass
	preds = model(device, data).to(device)
	assert not torch.any(torch.isnan(preds)), 'Model returned "NaN" as prediction'
	loss = loss_fn(preds, truth)
	assert not torch.any(torch.isnan(loss)), 'Model made numerical prediction, but loss was calculated as "NaN"'

	
	return truth, preds, loss

######################################
## Training & Validation for an Epoch
######################################
def train_epoch(training_data,
				validation_data, 
				model,
				optimizer,
				loss_fn,
				summary_dict: dict,
				train_sample_dict: dict,
				val_sample_dict: dict,
				device: torch.device):
	""" Function to complete a training step on a single sample

		Args:
			training_data (torch.dataloader): dataloader containing the training samples
			validation_data (torch.dataloader): dataloader containing the validation samples
			model (loaded pytorch model): model to train
			optimizer (torch.optim): optimizer for training set
			loss_fn (torch.nn Loss Function): loss function; Distribution_Loss for the groundwater data
			summary_dict (dict): dictionary to save epoch stats to
			train_sample_dict (dict): dictionary to save training sample stats to
			val_sample_dict (dict): dictionary to save validation sample stats to
			device (torch.device): device index to select

		Returns:
			summary_dict (dict): dictionary with epoch stats
			train_sample_dict (dict): dictionary with training sample stats
			val_sample_dict (dict): dictionary with validation sample stats
	"""
	## Initialize things to save
	startTime = time.time()
	trainbatches = len(training_data)
	valbatches = len(validation_data)
	trainbatch_ID = 0
	valbatch_ID = 0

	## Train on all training samples
	for traindata in training_data:
		trainbatch_ID += 1
		truth, pred, train_loss = train_datastep(traindata, 
													model,
													optimizer,
													loss_fn,
													device)
		train_sample_dict = append_to_dict(train_sample_dict, trainbatch_ID, truth, pred, train_loss)
	train_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

	## Evaluate on all validation samples
	with torch.no_grad():
		for valdata in validation_data:
			valbatch_ID += 1
			truth, pred, val_loss = eval_datastep(valdata, 
													model,
													loss_fn,
													device)
			val_sample_dict = append_to_dict(val_sample_dict, valbatch_ID, truth, pred, val_loss)
	val_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

	## Calcuate the Epoch Average Loss
	train_samples = train_batchsize * trainbatches
	val_samples = val_batchsize * valbatches
	avgTrainLoss = np.sum(train_sample_dict["loss"][-train_samples:]) / train_samples
	avgValLoss = np.sum(val_sample_dict["loss"][-val_samples:]) / val_samples
	summary_dict["train_loss"].append(avgTrainLoss)
	summary_dict["val_loss"].append(avgValLoss)

	## Calculate Time
	endTime = time.time()
	epoch_time = (endTime - startTime) / 60
	summary_dict["epoch_time"].append(epoch_time)

	return summary_dict, train_sample_dict, val_sample_dict