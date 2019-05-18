import argparse
import os 

import utils
import json

import fc_model 
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
                description='Uses a trained network loaded from the specified checkpoint to predict the class for an input image')

# --- Required arguments
parser.add_argument('image_path', action='store',
                   help='Path to an image.')
parser.add_argument('checkpoint_path', action='store',
                   help='Path to checkpoint.')

# --- Optional arguments
parser.add_argument('--top_k', action='store',
                    default=5,
                    type=int,
                    help='Top K most likely classes. Default is 5.')

parser.add_argument('--category_names', action="store",
                   default='cat_to_name.json',
                   help='File name of a mapping of categories to real names. Default is cat_to_name.json')

parser.add_argument('--gpu', action="store_true", 
                    default=False,
                   help='Use GPU for inference')

parser.add_argument('--show_plot', action="store_true", 
                    default=False,
                   help='Show plot of the top classes')

results = parser.parse_args()
# print('image_path         = {!r}'.format(results.image_path))
# print('checkpoint_path    = {!r}'.format(results.checkpoint_path))

# print('top_k            = {!r}'.format(results.top_k))
# print('category_names   = {!r}'.format(results.category_names ))
# print('gpu              = {!r}'.format(results.gpu))
# print('show_plot        = {!r}'.format(results.show_plot))


# --------------------------------

# Check to ensure that data_dir is present first before proceeding
image_path = results.image_path
assert os.path.exists(image_path), "image_path must be exist"

checkpoint_path = results.checkpoint_path
assert os.path.exists(checkpoint_path), "checkpoint_path must be exist"


# Load cat_to_name
with open(results.category_names, 'r') as f:
    cat_to_name = json.load(f)

# --------------------------------

# --- Create model from the checkpoint data
data_checkpoint = torch.load(checkpoint_path)

network = fc_model.Network(
    data_checkpoint['arch'], 
    cat_to_name,
    hidden_units = data_checkpoint['hidden_units'],
    gpu=results.gpu)

loaded_model = network.model
loaded_model.load_state_dict(data_checkpoint['state_dict'])

# Show prediction info
utils.view_classify(
    image_path, 
    loaded_model, 
    cat_to_name, 
    data_checkpoint['class_to_idx'], 
    topk=results.top_k, 
    show_plot=results.show_plot)
