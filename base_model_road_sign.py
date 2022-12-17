#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pytorch imports

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


# In[2]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, PIL
from glob import glob
import tensorflow as tf
from io import StringIO 
from PIL import Image

from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

#import seaborn as sns
#from sklearn import model_selection


import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import  img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#from tqdm import tqdm


# In[3]:


""" Sequential Model Architecture """
Sequential = tf.keras.models.Sequential

""" Data Preprocessing Functions """
Resizing = tf.keras.layers.experimental.preprocessing.Resizing
Rescaling = tf.keras.layers.experimental.preprocessing.Rescaling

""" Data Augmentation Functions """
RandomFlip = tf.keras.layers.experimental.preprocessing.RandomFlip
RandomRotation = tf.keras.layers.experimental.preprocessing.RandomRotation
RandomZoom = tf.keras.layers.experimental.preprocessing.RandomZoom

""" Artificial Neural Network Layer Inventory """
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

""" Convolutional Neural Network Layer Inventory """
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
Flatten = tf.keras.layers.Flatten

""" Residual Network Layer Inventory """
ResNet50 = tf.keras.applications.resnet50.ResNet50

""" Function to Load Images from Target Folder """
get_image_data = tf.keras.utils.image_dataset_from_directory


# In[4]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 128, 128


# In[5]:


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transformations = {
    'train': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.CenterCrop((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}


# In[6]:


learning_rate = 0.001
batch_size = 4
num_epochs = 8
num_classes = 8

# device
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(device)


# In[7]:


DIR_PATH='D:\\road sign\\'


# In[8]:


total_dataset = torchvision.datasets.ImageFolder(DIR_PATH,transform=transformations['train'])

len(total_dataset),total_dataset[0][0].shape,total_dataset.class_to_idx


# In[9]:


train ='D:\\road sign\\train\\'
test ='D:\\road sign\\test\\'
nb_train_samples =400
nb_validation_samples = 100
epochs = 10
batch_size = 4


# In[10]:


train = torchvision.datasets.ImageFolder(train,transform=transformations['train'])

len(train),train[0][0].shape,train.class_to_idx


# In[11]:


test = torchvision.datasets.ImageFolder(test,transform=transformations['test'])

len(test),test[0][0].shape,test.class_to_idx


# In[12]:


# dataloaders
train_loader = DataLoader(dataset=train,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=4)


# In[13]:


print(len(train_loader))
#print(len(val_loader))
print(len(test_loader))


# In[14]:


# testing dataloading 

examples = iter(train_loader)
samples,labels = examples.next()
print(samples.shape,labels.shape) # batch_size=4
len(train_loader)
#len(val_loader)


# In[16]:


import time
import numpy as np
import os
#import path

from typing import List, Tuple
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import PIL.Image
import pathlib
import shutil

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model

from tensorflow.python.keras.utils import layer_utils
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG

import scipy.misc

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last') # can be channels_first or channels_last. 
K.set_learning_phase(1) # 1 stands for learning phase


# In[17]:


import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# In[26]:


class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)


# In[18]:


# custom CNN model class

class ConvNet(nn.Module):
    def __init__(self,model,num_classes):
        super(ConvNet,self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1]) # model excluding last FC layer
        self.linear1 = nn.Linear(in_features=2048,out_features=512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512,out_features=num_classes)
    
    def forward(self,x):
        x = self.base_model(x)
        x = torch.flatten(x,1)
        lin = self.linear1(x)
        x = self.relu(lin)
        out = self.linear2(x)
        return lin, out


# In[19]:


#model = torchvision.models.VGG16(pretrained=True)# base model
model = torchvision.models.resnet50(pretrained=True) # base model

model = ConvNet(model,num_classes)
#model = SEBottleneck()
#model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
print(model)


# In[20]:


num_epochs=1


# In[21]:


print(len(train_loader))


# In[22]:


# training loop

n_iters = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    for ii,(images,labels) in enumerate(train_loader):
        print(ii)
        images = images.to(device)
        labels = labels.to(device)
        
        _,outputs = model(images)
        loss = criterion(outputs,labels)
        
        # free_gpu_cache()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _,preds = torch.max(outputs, 1)
        #running_loss += loss.item() * images.size(0)
        #running_corrects += torch.sum(preds == labels.data)
        
        if (ii+1)%108 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{ii+1}/{n_iters}], Loss = {loss.item():.6f}')
            
    print('----------------------------------------')
    


# In[23]:


# evaluating model and getting features of every image

def eval_model_extract_features(features,true_labels,model,dataloader,phase):

    with torch.no_grad():
        # for entire dataset
        n_correct = 0
        n_samples = 0

        model.eval()

        for images,labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            true_labels.append(labels)
            
            ftrs,outputs = model(images)
            features.append(ftrs)

            _,preds = torch.max(outputs,1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
                
        accuracy = n_correct/float(n_samples)

        print(f'Accuracy of model on {phase} set = {(100.0 * accuracy):.4f} %')

    return features,true_labels


# In[24]:


features = []
true_labels = []


# In[25]:


train_loader = DataLoader(dataset=train,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)

features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=train_loader,phase='training')

print(len(features),len(true_labels))


# In[27]:


features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=test_loader,phase='validation')

print(len(features),len(true_labels))


# In[ ]:




