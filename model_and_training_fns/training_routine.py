# TRAINING SCRIPT FOR CSIRO GROUNDWATER MODELS
"""

"""
#############################################
## Packages
#############################################
import sys
import os
import psutil
import typing
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath('./'))
from model_definition import CNN
import training_fns as tr
from dataloader import GW_DataSet, make_dataloader
from continue_study import continuation_setup

#############################################
## Inputs
#############################################
descr_str = 'Trains a CNN on the CSIRO groundwater dataset'
parser = argparse.ArgumentParser(prog='CSIRO Groundwater Training',
								 description=descr_str,
								 fromfile_prefix_chars='@')

#############################################
## Learning Problem
#############################################
parser.add_argument('--studyIDX',
					action='store',
					type=int,
					default=1,
					help='Study ID number to match hyperparameters')

#############################################
## File Paths
#############################################
parser.add_argument('--data_dir',
					action='store',
					type=str,
					default='../NPYdata',
					help='Directory with "train", "tune", and "test" subdirectories, containing .npy files')

#############################################
## Model Parameters
#############################################
parser.add_argument('--local_effects_kernel',
					action='store',
					type=int,
					default=3,
					help='Size of the 2D convolutional kernel in the Local Effects Modules')

parser.add_argument('--local_effects_neurons',
					action='store',
					type=int,
					default=8,
					help='Number of neurons (equivalently "features" or "channels") in the Local Effects Modules')

parser.add_argument('--local_effects_bias',
					action='store',
					type=bool,
					default=True,
					help='If the 2D convolutions learn bias in the Local Effects Modules')

parser.add_argument('--local_effects_depth',
					action='store',
					type=int,
					default=3,
					help='How many sets of [Conv2D, Activation] pairs before flatteing in the Local Effects Modules')

parser.add_argument('--activation',
					action='store',
					type=str,
					default='nn.ReLU',
					help='Torch nn.modules.activation layer to use as a activation function')

parser.add_argument('--dense_neurons',
					action='store',
					type=int,
					default=32,
					help='Number of neurons (equivalently "features" or "channels") in the dense layers after the Local Effects Modules')

parser.add_argument('--dense_bias',
					action='store',
					type=bool,
					default=True,
					help='If the dense layers after the Local Effects Modules learn bias')

parser.add_argument('--dense_depth',
					action='store',
					type=int,
					default=4,
					help='How many dense layers (including the final prediction layer) are included after the Local Effects Moduls')

#############################################
## Training Parameters
#############################################
parser.add_argument('--init_learnrate',
					action='store',
					type=float,
					default=1e-6, #Value used in Dan's paper
					help='Initial learning rate')

parser.add_argument('--batch_size',
					action='store',
					type=int,
					default=64,
					help='Batch size')

#############################################
## Epoch Parameters
#############################################
parser.add_argument('--total_epochs',
					action='store',
					type=int,
					default=10,
					help='Total training epochs')

parser.add_argument('--cycle_epochs',
					action='store',
					type=int,
					default=5,
					help='Number of epochs between saving the model and re-quequing training process; must be able to be completed in the set wall time')

parser.add_argument('--train_batches',
					action='store',
					type=int,
					default=250,
					help='Number of batches to train on in a given epoch')

parser.add_argument('--val_batches',
					action='store',
					type=int,
					default=25,
					help='Number of batches to validate on in a given epoch')

parser.add_argument('--continuation',
					action='store_true',
					help='Indicates if training is being continued or restarted')

parser.add_argument('--checkpoint',
					action='store',
					type=str,
					default='None',
					help='Path to checkpoint to continue training from')


#############################################
#############################################
if __name__ == '__main__':

	#############################################
	## Process Inputs
	#############################################
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()

	## Study ID
	studyIDX = args.studyIDX

	## Data Paths
	data_dir = args.data_dir

	## Model Parameters
	LE_kernel = args.local_effects_kernel
	LE_neurons = args.local_effects_neurons
	LE_bias = args.local_effects_bias
	LE_depth = args.local_effects_depth
	activation = eval(args.activation)
	dense_neurons = args.dense_neurons
	dense_bias = args.dense_bias
	dense_depth = args.dense_depth

	## Training Parameters
	initial_learningrate = args.init_learnrate
	batch_size = args.batch_size

	## Epoch Parameters
	total_epochs = args.total_epochs
	cycle_epochs = args.cycle_epochs
	train_batches = args.train_batches
	val_batches = args.val_batches
	CONTINUATION = args.continuation
	START = not CONTINUATION
	checkpoint = args.checkpoint

	#############################################
	## Check Devices
	#############################################
	print('\n')
	print('Slurm & Device Information')
	print('=========================================')
	print('Slurm Job ID:', os.environ['SLURM_JOB_ID'])
	print('Pytorch Cuda Available:', torch.cuda.is_available())
	print('GPU ID:', os.environ['SLURM_JOB_GPUS'])
	print('Number of System CPUs:', os.cpu_count())
	print('Number of CPUs per GPU:', os.environ['SLURM_JOB_CPUS_PER_NODE'])

	print('\n')
	print('Model Training Information')
	print('=========================================')

	#############################################
	## Initialize Model
	#############################################
	model = CNN(local_effects_kernel = LE_kernel,
				local_effects_neurons = LE_neurons,
				local_effects_bias = LE_bias,
				local_effects_depth = LE_depth,
				activation = activation,
				dense_neurons = dense_neurons,
				dense_bias = dense_bias,
				dense_depth = dense_depth)

	#############################################
	## Initialize Optimizer
	#############################################
	# all parameters other than learning_rate are pytorch defaults
	optimizer = torch.optim.RMSprop(model.parameters(),
									lr = initial_learningrate, 
									alpha=0.9, #same as Dan's paper & keras default
									eps=1e-08, #all other parameters are pytorch default
									weight_decay=0, 
									momentum=0, 
									centered=False, 
									foreach=None, 
									maximize=False, 
									differentiable=False)

	#############################################
	## Initialize Loss
	#############################################
	# custom loss function as defined in Pagendam et al. 2023
	# v2 is a differnt implementation of the same function that we hope is more stable
	loss_fn = tr.Distribution_Loss_v2().to(device)

	print('Model initialized.')

	#############################################
	## Load Model for Continuation
	#############################################
	if CONTINUATION:
		model.to(device)
		checkpoint = torch.load(checkpoint, map_location=device)
		starting_epoch = checkpoint["epoch"]
		model.load_state_dict(checkpoint["modelState"])
		optimizer.load_state_dict(checkpoint["optimizerState"])
		print('Model state loaded for continuation.')
	else:
		starting_epoch = 0

	#############################################
	## Initialize Data
	#############################################
	train_dataset = GW_DataSet(os.path.join('..', data_dir, 'train'))
	val_dataset = GW_DataSet(os.path.join('..', data_dir, 'tune'))
	test_dataset = GW_DataSet(os.path.join('..', data_dir, 'test'))

	assert train_batches*batch_size <= train_dataset.__len__(), "{} training batches of batchsize {} is larger than training dataset of size {}; adjust training parameters.".format(train_batches, batch_size, train_dataset.__len__())
	assert val_batches*batch_size <= val_dataset.__len__(), "{} validation batches of batchsize {} is larger than validation dataset of size {}; adjust validation parameters.".format(val_batches, batch_size, val_dataset.__len__())

	print('Datasets initialized.')

	#############################################
	## Training Loop
	#############################################
	##Initialize Dictionaries
	train_val_summary_dict = {
						"train_loss": [],
						"val_loss": [],
						"epoch_time": []
						}
	train_samples_dict = {
						"epoch": [],
						"batch": [],
						"truth": [],
						"mu1_prediction": [],
						"mu2_prediction": [],
						"log_sigma1_prediction": [],
						"log_sigma2_prediction": [],
						"loss": []
						}
	val_samples_dict = {
						"epoch": [],
						"batch": [],
						"truth": [],
						"mu1_prediction": [],
						"mu2_prediction": [],
						"log_sigma1_prediction": [],
						"log_sigma2_prediction": [],
						"loss": []
						}

	## Train Model
	print("Training Model . . .")
	starting_epoch += 1
	ending_epoch = min(starting_epoch+cycle_epochs, total_epochs+1)

	for e in range(starting_epoch, ending_epoch):
		## Setup Dataloaders
		train_dataloader = make_dataloader(train_dataset, batch_size, train_batches)
		val_dataloader = make_dataloader(val_dataset, batch_size, val_batches)

		## Train an Epoch
		train_val_summary_dict, train_samples_dict, val_samples_dict = tr.train_epoch(training_data = train_dataloader,
																						validation_data = val_dataloader, 
																						model = model,
																						optimizer = optimizer,
																						loss_fn = loss_fn,
																						summary_dict = train_val_summary_dict,
																						train_sample_dict = train_samples_dict,
																						val_sample_dict = val_samples_dict,
																						device = device)

		## Add Epoch Info to Dicts
		#### this works by replacing "epoch zero" with the correct epoch
		train_samples_dict["epoch"] = [x or e for x in train_samples_dict["epoch"]]
		val_samples_dict["epoch"] = [x or e for x in val_samples_dict["epoch"]]

		## Print Summary Results
		print('Completed epoch '+str(e)+':')
		print('\tTraining Loss:', train_val_summary_dict["train_loss"][-1])
		print('\tValidation Loss:', train_val_summary_dict["val_loss"][-1])
		print('\tEpoch Time:', train_val_summary_dict["epoch_time"][-1])

	## Save Model Checkpoint
	print("Saving model checkpoint at end of epoch "+ str(e) + ". . .")
	new_checkpoint = {
					"epoch"          : e,
					"modelState"     : model.state_dict(),
					"optimizerState" : optimizer.state_dict()
					}
	new_checkpoint_path = os.path.join('./', 'study{0:02d}_modelState_epoch{1:03d}.pth'.format(studyIDX, e))
	torch.save(new_checkpoint, new_checkpoint_path)
	print("Model checkpoint saved at end of epoch "+ str(e) + ".")

	## Save Summary Dictionary
	summarydf = pd.DataFrame.from_dict(train_val_summary_dict, orient='columns')
	summarydf['epochs_index'] = np.arange(starting_epoch, ending_epoch)
	summarydf.set_index('epochs_index', drop=True, append=False, inplace=True)
	summary_csv_path = os.path.join('./', 'study{0:02d}_trainval_summary.csv'.format(studyIDX))
	tr.save_append_df(summary_csv_path, summarydf, START)

	## Save Sample Dictionaries
	trainsamplesdf = pd.DataFrame.from_dict(train_samples_dict, orient='columns')
	trainsamples_csv_path = os.path.join('./', 'study{0:02d}_train_samples.csv'.format(studyIDX))
	tr.save_append_df(trainsamples_csv_path, trainsamplesdf, START)
	valsamplesdf = pd.DataFrame.from_dict(val_samples_dict, orient='columns')
	valsamples_csv_path = os.path.join('./', 'study{0:02d}_val_samples.csv'.format(studyIDX))
	tr.save_append_df(valsamples_csv_path, valsamplesdf, START)

	print("Training and Validation results sucessfully written to csv.")

	#############################################
	## Continue if Necessary
	#############################################
	FINISHED_TRAINING = e+1 > total_epochs
	if not FINISHED_TRAINING:
		new_slurm_file = continuation_setup(new_checkpoint_path, studyIDX, last_epoch=e)
		os.system(f'sbatch {new_slurm_file}')

	#############################################
	## Run Test Set When Training is Complete
	#############################################
	if FINISHED_TRAINING:
		print("Testing Model . . .")
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
		testbatch_ID = 0
		testing_dict = {
						"epoch": [],
						"batch": [],
						"truth": [],
						"mu1_prediction": [],
						"mu2_prediction": [],
						"log_sigma1_prediction": [],
						"log_sigma2_prediction": [],
						"loss": []
						}

		with torch.no_grad():
			for testdata in test_dataloader:
				testbatch_ID += 1
				truth, pred, loss = tr.eval_datastep(testdata, 
														model,
														loss_fn,
														device)
				testing_dict = tr.append_to_dict(testing_dict, testbatch_ID, truth, pred, loss)


		## Save Testing Info
		del testing_dict["epoch"]
		testingdf = pd.DataFrame.from_dict(testing_dict, orient='columns')
		testingdf.to_csv(os.path.join('./', 'study{0:02d}_testset_results.csv'.format(studyIDX)))
		print('Model testing results saved.')

		print('STUDY{0:02d} COMPLETE'.format(studyIDX))
		print('\n')