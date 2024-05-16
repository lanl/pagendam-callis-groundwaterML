[//]: <> (THIS IS A MARKDOWN FILE, VIEW IN A MARKDOWN VIEWER OR CONVERT)

# PyTorch Implementation of Log-Additive Convolutional Neural Networks

This repository contains an implementation of Log-Additive Convolutional Neural Networks in Pytorch, following the model architecture described in [*A log-additive neural model for spatio-temporal prediction of groundwater levels*](https://doi.org/10.1016/j.spasta.2023.100740) by Pagendam et. al. 

These models are intended to be trained on the groundwater data available at [https://doi.org/10.25919/skw8-yx65](https://doi.org/10.25919/skw8-yx65). Data must be convered from `.rds` files to `.npy` files to be compatible with this code. Code to complete this conversion can be found in `/notebooks`.

The repository does include randomly generated synthetic data. This synthetic data has the same shapes as the original groundwater data. However, all values were randomly generated. Therefore, this data does not present a substitute learning problem. It is included so users can have an example of how data is loaded into the PyTorch DataSet and DataLoader for model training.

This code was developed by **Skylar Callis** while working as a post-bachelors student at [Los Alamos National Laboratory (LANL)](https://www.lanl.gov/?source=globalheader) from 2022 - 2024. To see what they are up to these days, visit [Skylar's Website ](https://skylar-jean.com). 

This work was done in collaboration with Dan Pagendam, the Team Leader for Hybrid Modelling in Analytics & Decision Sciences in Data61 at CSIRO.

The PyTorch Implementation of Log-Additive Convolutional Neural Networks code has been approved by LANL for a BSD-3 open source license under O#4752. 

The GitHub page for this code can be found [here](https://github.com/lanl/pagendam-callis-groundwaterML).

## Model and Training Functions

This directory contains the definitions of the PyTorch models, datasets, and dataloaders used in this learning problem. It also contains functions that define the training steps, as well as a script that completes the entire training routine.

The `model_definition.py` and `dataloader.py` files have unit tests, found in the `model_and_training_fns/tests` directory. These are implemented in `pytest`, and can be executed by running `pytest` in the terminal. To see coverage, run `pytest --cov`.

## Notebooks

The notebooks directory contains all Jupyter Notebooks. Notebooks included are:

 - `generate_synth_data.ipynb`: generates synthetic data in the same shape as the groundwater data from [https://doi.org/10.25919/skw8-yx65](https://doi.org/10.25919/skw8-yx65).
 - `convert_rsd_npy.ipynb`: converts the `.rsd` files found at [https://doi.org/10.25919/skw8-yx65](https://doi.org/10.25919/skw8-yx65) to `.npy` files.
 - `plot_data.ipynb`: plots example datapoints from [https://doi.org/10.25919/skw8-yx65](https://doi.org/10.25919/skw8-yx65).
 - `models.ipnb`: develops models that follow [*A log-additive neural model for spatio-temporal prediction of groundwater levels*](https://doi.org/10.1016/j.spasta.2023.100740) by Pagendam et. al. 
 - `training_test_run.ipynb`: completes a run-through of the training routine, using the methodology from `model_and_training_fns/training_routine.py`, supported by `model_and_training_fns/training_fns.py`; additionally, describes issues that arose regarding the RMSProp Optimizer and the Distribution Loss function. It also attempts to debug why the models aren't learning effectively by performing checks on the training functions.
 - `model_reslts.ipynb`: plots the results of the trained models
 
 ## Synthetic Data
 
 This directory contains randomly generated data files of the same shape as the data in [https://doi.org/10.25919/skw8-yx65](https://doi.org/10.25919/skw8-yx65). It contains subdirectories for `train`, `tune` (validation), `test` data. All of the data is randomly generated values $\in [0, 1)$.