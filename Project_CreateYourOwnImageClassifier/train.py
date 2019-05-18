import argparse
import os 
import json
import time

import fc_model 
import torch


dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
                description='Train a new network on a dataset and save the model as a checkpoint')

# --- Required arguments

parser.add_argument('data_dir', action='store',
                   help='Directory to the Train data.')

# --- Optional arguments

parser.add_argument('--save_dir', action="store",
                   default=os.path.join(dir_path, 'checkpoint'),
                   help='Directory to save checkpoints. Default is ./checkpoint')
parser.add_argument('--arch', action='store',
                    default='vgg13',
                    help='Architecture - must be either "alexnet" or "vgg*". Default is "vgg13".')
parser.add_argument('--category_names', action="store",
                   default='cat_to_name.json',
                   help='File name of a mapping of categories to real names. Default is cat_to_name.json')

# Hyperparameters
parser.add_argument('--learning_rate', action='store',
                    default=0.01,
                    type=float,
                    help='Hyperparameter - Learning rate. Default is 0.01.')
parser.add_argument('--hidden_units', action='store',
                    default=512,
                    type=int,
                    help='Hyperparameter - Number of hidden units. Default is 512.')
parser.add_argument('--epochs', action='store',
                    default=10,
                    type=int,
                    help='Hyperparameter - Number of epochs. Default is 10.')
parser.add_argument('--gpu', action="store_true", 
                    default=False,
                   help='Use GPU for training')

results = parser.parse_args()

# --------------------------------
# Check to ensure that data_dir is present first before proceeding
data_dir = results.data_dir
assert os.path.exists(data_dir), "data_dir must be exist"

# Load cat_to_name
with open(results.category_names, 'r') as f:
    cat_to_name = json.load(f)

#---------------------------------

network = fc_model.Network(
    results.arch, 
    cat_to_name,
    hidden_units = results.hidden_units,
    gpu=results.gpu)

network.train(
            results.data_dir,
            learning_rate = results.learning_rate,
            epochs = results.epochs)


# Save the checkpoint
checkpoint = {
              'arch':           results.arch,
              'data_dir':       results.data_dir,
              'cat_to_name':    cat_to_name,
              'gpu':            results.gpu,
              'hidden_units':   results.hidden_units,
              'epochs':         results.epochs,
              'learning_rate':  results.learning_rate,
              'class_to_idx':   network.class_to_idx,
              'state_dict':     network.model.state_dict()}

save_dir = results.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

checkpoint_name = 'checkpoint_{}.pth'.format(time.strftime("%Y%m%d_%H%M%S"))
checkpoint_path = os.path.join(save_dir, checkpoint_name)

torch.save(checkpoint, checkpoint_path)
print('* Checkpoint saved to : {}'.format(checkpoint_path))
    
    
#---------------------------------
