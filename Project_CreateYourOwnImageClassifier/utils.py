import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Load image using PIL and create a copy of it 
    # to ensure that we don't modify the original model
    pil_im = Image.open(image, 'r').copy()
    
    # 1) Resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    pil_im = pil_im.resize((256, 256))
    
    # 2) Crop out the center 224x224 portion of the image.
    width, height = pil_im.size   # Get dimensions

    new_width = 224
    new_height = 224

    # Get a location in the image that will be used to crop the image
    
    # # --- Horizontal dimestion 
    # if the width is less than 224, do nothing. 
    # Otherwise, compute the left & right position
    
    left = 0
    right = width
    
    if (width > new_width):
        left = (width - new_width)//2
        right = left + new_width
        
    # # --- Vertical dimestion 
    # if the height is less than 224, do nothing. 
    # Otherwise, compute the top & bottom position

    top = 0
    bottom = height
    
    if (height > new_height):
        top = (height - new_height)//2
        bottom = top + new_height
 
    new_pil = pil_im.crop((left, top, right, bottom))
    
    # 3) Convert the color channels to be in a range of float 0-1
    # since the model expects this range
    np_image = np.array(new_pil)/255.
    
    # 4) Normalize the image in such a way that expects by the model
    normalized_im = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # 5) PyTorch expects the color channel to be the first dimension but it's the 
    # third dimension in the PIL image and Numpy array. 
    # Hence, we need to reorder the dimensions before returning the image value
    new_normalized_im = normalized_im.transpose(2, 0, 1)
    
    return new_normalized_im

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if (title):
        ax.set_title(title)
        
    ax.imshow(image)
    
    return ax

def predict(img_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img_np = process_image(img_path)
    img = torch.from_numpy(img_np)

    # The loaded image does not have a batch dimension and we need to add
    # it to the torch image data since it is expected by our model
    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    img.unsqueeze_(0)

    model.eval()

    with torch.no_grad():
        logits = model.forward(img.to(dtype=torch.float))

    # Get the class probabilities
    ps = F.softmax(logits, dim=1)
    
    # Get the top probabilities and their class index
    probs, classes = ps.topk(topk)
    
    # convert data to numpy data befor return them
    return probs[0].numpy(), classes[0].numpy()

def view_classify(img_path, model, cat_to_name, class_to_idx, topk=5, show_plot=False):
    ''' This function predicts the class (or classes) of an image using 
    the specified trained deep learning model and display the top k classes
    from prediciton
    '''
    
    # Get the top probabilities and their class index after predicting a given img_path
    probs, classe_indice = predict(img_path, model, topk=topk)
    
    # Get the top class names
    top_class = []

    for cur_class_index in classe_indice:
        for cat_index, index in class_to_idx.items():   
            if index == cur_class_index:
                top_class.append(cat_to_name[cat_index])
    
    print('Prediction: {}'.format(top_class[0]))
    print('Top classes:');
   
    for (cur_index, cur_class, cur_prob) in zip(range(topk), top_class, probs): 
        print('    ({}) {} : {:4f}'.format(cur_index+1, cur_class, cur_prob))
    
    
    # Show a chart when the showPlot value is set to True
    if (show_plot):

        # Create a plot to display the image and a chart of top 5 classes getting after
        # the prediciton
        fig, (ax1, ax2) = plt.subplots(figsize=(5,7), nrows=2)

        # --- 1st subplot: Display the image
        img_val = process_image(img_path)
        imshow(img_val, ax=ax1, title=top_class[0])
        ax1.axis('off')

        # --- 2nd subplot: Top 5 classes

        bar_locations = np.arange(5)
        ax2.barh(bar_locations, probs)
        ax2.set_yticks(bar_locations)
        ax2.set_yticklabels(top_class)
        ax2.set_xlim(0, 1.1)

        plt.gca().invert_yaxis()

        plt.tight_layout()