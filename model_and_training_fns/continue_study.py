""" Function to continue a study from a checkpoint
"""
####################################
## Packages
####################################
import os
import sys

####################################
## Function
####################################
def continuation_setup(checkpointpath, studyIDX, last_epoch):
    """ Function to generate the training.input and training.slurm files for continuation of model training
        Args:
            checkpointpath (str): path to model checkpoint to load in model from
            studyIDX (int): study ID to include in file name
            last_epoch (int): numer of epochs completed at this checkpoint
        Returns:
            new_training_slurm_filepath (str): name of slurm file to submit job for continued training
    """
    ## Identify Template Files
    training_input_tmpl = "./training_input.tmpl"
    training_slurm_tmpl = "./training_slurm.tmpl"

    ## Make new training.input file
    with open(training_input_tmpl, 'r') as f:
        training_input_data = f.read()
    new_training_input_data = training_input_data.replace('<CHECKPOINT>', checkpointpath)        
    new_training_input_filepath = 'study{0:02d}_restart_training_epoch{1:03d}.input'.format(studyIDX, last_epoch+1)
    with open(os.path.join('./', new_training_input_filepath), 'w') as f:
        f.write(new_training_input_data)

    with open(training_slurm_tmpl, 'r') as f:
        training_slurm_data = f.read()
    new_training_slurm_filepath = 'study{0:02d}_restart_training_epoch{1:03d}.slurm'.format(studyIDX, last_epoch+1)
    new_training_slurm_data = training_slurm_data.replace('<INPUTFILE>', new_training_input_filepath)
    new_training_slurm_data = new_training_slurm_data.replace('<epochIDX>', '{0:03d}'.format(last_epoch+1))         
    with open(os.path.join('./', new_training_slurm_filepath), 'w') as f:
        f.write(new_training_slurm_data)

    return new_training_slurm_filepath