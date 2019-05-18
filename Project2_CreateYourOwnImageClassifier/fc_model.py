import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
import datetime
import time

class Network():
    
    def __init__(self, arch, cat_to_name, hidden_units=512, gpu=False):
        
        self.arch           = arch
        self.cat_to_name    = cat_to_name
        self.hidden_units   = hidden_units
        self.gpu            = gpu

        self._initModel()
        
    def train(self, data_dir, learning_rate=0.01, epochs=20):
        
        self.data_dir       = data_dir
        self.learning_rate  = learning_rate
        self.epochs         = epochs
        
        self._setupModel()
        
        print('**********************')
        print('--- Model Settings ---')
        print('**********************')
        print('* Train data:                {}'.format(self.data_dir))
        print('* Pre-trained network:       {}'.format(self.arch))
        print('* Nunmber of hidden units:   {}'.format(self.hidden_units))
        print('* Learning rate:             {}'.format(self.learning_rate))
        print('* Nunmber of epochs:         {}'.format(self.epochs))
        print('* Mode:                      {}'.format(self.device))
        print('**********************\n')
        
        print("[{}] Start training ".format(datetime.datetime.now()))
        
        print_every = 20
        steps = 0
        
        self.model.to(self.device) 
        for e in range(epochs):
            
            self.model.train()
            
            running_loss = 0
            for ii, (inputs, labels) in enumerate(self.testdataloaders):
                steps += 1
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()

                # Forward and backward passes
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    
                    # Make sure network is in eval mode for inference
                    self.model.eval()
                    
                    dataloader = self.validationdataloaders
                    
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self.validation(dataloader)

                    print("[{}] ".format(datetime.datetime.now()),
                          "Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(dataloader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(dataloader)))

                    running_loss = 0
                    
                    self.model.train()
                    
        print("[{}] Finish training ".format(datetime.datetime.now()))

    def _initModel(self):
    
        # 1) Load a pre-trained network. 
        strCreateModel = 'models.{}(pretrained=True)'.format(self.arch)
        self.model = eval(strCreateModel)

        for param in self.model.parameters():
            param.requires_grad = False
            
        # Some network (e.g. Alexnet) has a Drop out as the first layer
        # and we need to go to the next layer to find a number of in_features
            
        # Create a new classifier and assign it to the pre-trained model
        # Input to a hidden layer
        
        for index in range(len(self.model.classifier)):
            first_layer = self.model.classifier[index]
            
            
            if 'in_features' in first_layer.__dict__.keys():
                self.input_size = first_layer.in_features
                break
              
        self.output_size = len(self.cat_to_name)
        
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_size, self.hidden_units)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.hidden_units, self.output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        self.model.classifier = classifier
        
            
        # Set the model to cuda mode if gpu is set to True
        if (self.gpu):
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device) 

    def _initTransform(self): 
        
        # Transform - Test data
        data_transforms_test = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
        
        image_datasets_test = datasets.ImageFolder(
                                        self.data_dir + '/test', 
                                        transform=data_transforms_test)
        self.testdataloaders = torch.utils.data.DataLoader(image_datasets_test, batch_size=64, shuffle=True)
        
        # Transform - Validation data
        
        data_transforms_validation = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
        
        image_datasets_validation = datasets.ImageFolder(
                                        self.data_dir + '/valid', 
                                        transform=data_transforms_validation)
        
        self.validationdataloaders = torch.utils.data.DataLoader(
                                        image_datasets_validation, batch_size=32)
        
        # --------
        self.class_to_idx = image_datasets_test.class_to_idx  
        
        
    def _setupModel(self):
        
        # Create a transform
        self._initTransform()
         
        self.criterion = nn.NLLLoss()

        # The hyper-parameters below got from the tuning step above
        self.optimizer = optim.Adam(self.model.classifier.parameters(), 
                                    lr=self.learning_rate)
        
    def validation(self, dataloader):
        
        self.model.to(self.device) 
        
        # Make sure network is in eval mode for inference
        self.model.eval()
            
        test_loss = 0
        accuracy = 0
        for images, labels in dataloader:

            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model.forward(images)
            test_loss += self.criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy
        
