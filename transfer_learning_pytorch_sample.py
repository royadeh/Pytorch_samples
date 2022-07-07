# -*- coding: utf-8 -*-
"""Classification_Pipeline.ipynb


Import package
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
import os
import time
import copy

"""Dataset for train and validation
the dataset is 800 images of cat and dog
you can change it to your dataset """

#download the dataset from google drive
#! gdown https://drive.google.com/file/d/1AZs9AOsw5pmHGPDmO56yYNSdmTFXCQbx/view?usp=sharing

#or you can access your dataset in your google drive
from google.colab import drive
drive.mount('/content/drive')

"""DataLoader"""

#dataloader
def get_dataloaders(data, batch_size=80, shuffle=True):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      #Data Augmentation
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1)    
                               ]),
        'val': transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]),
    }


    data_dir= data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle)
                     for x in ['train','val']}
   
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders,image_datasets,dataset_sizes

"""this code was ran on google colab
you can download dataset and change this path to where the dataset has been downloaded"""

data= '/content/drive/MyDrive/train_val_pet'
dataloaders,image_datasets,dataset_sizes=get_dataloaders(data, batch_size=80, shuffle=True)

"""Pretrained model(alexnet)
Transfer Learning(change the last layer)
"""
def fine_tuning():
  model=models.alexnet(pretrained=True)
  #freeze features layers
  for param in model.features.parameters():
    param.requires_grad = False
  #the output of the previous layer is the input of the last layer
  #create the last leyer
  n_inputs = model.classifier[6].in_features
  last_layer = nn.Linear(n_inputs, len(image_datasets['train'].classes))
  #replace the last layer
  model.classifier[6] = last_layer
  print(model.classifier[6].out_features)
  return model




def train_model(model):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

  epochs = 2
  running_loss_history = []
  running_corrects_history = []
  val_running_loss_history = []
  val_running_corrects_history = []

  for epoch in range(epochs):
    
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    
    #model.train()
    for inputs, labels in dataloaders['train']:
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      #set it zero
      optimizer.zero_grad()
      #deriviative of loss
      loss.backward()
      #update the weights towards the lowest loss
      optimizer.step()
      
      _, preds = torch.max(outputs, 1)
      running_loss += loss.item()
      running_corrects += torch.sum(preds == labels.data)
      
    else:
      #model.evel()
      with torch.no_grad():
        for val_inputs, val_labels in dataloaders['val']:
          val_outputs = model(val_inputs)
          val_loss = criterion(val_outputs, val_labels)
          
          _, val_preds = torch.max(val_outputs, 1)

          val_running_loss += val_loss.item()
          val_running_corrects += torch.sum(val_preds == val_labels.data)
        
      epoch_loss = running_loss/len(dataloaders['train'].dataset)
      epoch_acc = running_corrects.float()/ len(dataloaders['train'].dataset)
      
      

      running_loss_history.append(epoch_loss)
      running_corrects_history.append(epoch_acc)
      
      val_epoch_loss = val_running_loss/len(dataloaders['val'].dataset)
      val_epoch_acc = val_running_corrects.float()/ len(dataloaders['val'].dataset)
      val_running_loss_history.append(val_epoch_loss)
      val_running_corrects_history.append(val_epoch_acc)
      print('epoch :', (epoch+1))
      print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
      print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
    return model,running_corrects_history,val_running_corrects_history,running_loss_history,val_running_loss_history

model=fine_tuning()
tuned_mode,running_corrects_history,val_running_corrects_history,running_loss_history,val_running_loss_history=train_model(model)



"""Matplotlib(Vidualization)"""

plt.plot(running_corrects_history, label='training accuracy')
plt.plot(val_running_corrects_history, label='validation accuracy')
plt.legend()

plt.plot(running_loss_history, label='training loss')
plt.plot(val_running_loss_history, label='validation loss')
plt.legend()