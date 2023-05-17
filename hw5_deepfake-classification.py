
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.optim as optim

from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random

from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


### xception model define 
# ref url: https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
#__all__ = ['xception']

model_urls = {
    'xception':'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
}

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
        
class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def xception(pretrained=False,**kwargs):
    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model
### xception model define end 


### train, test, print_confusion_matrix, plot function define 
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
epoch_t_labels = []
epoch_t_preds = []

def train(epoch):
  model.train() #!

  running_loss, running_corrects, num_cnt = 0.0, 0, 0
  
  for inputs, labels in dataloaders['train']:
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)
      num_cnt += len(labels)
                
  epoch_loss = float(running_loss / num_cnt)
  epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
  train_loss.append(epoch_loss)
  train_acc.append(epoch_acc)
  print('#epoch: {} train loss: {:.2f} acc: {:.1f}'.format(epoch, epoch_loss, epoch_acc))

def test(epoch, phase):
  model.eval() #!
  
  running_loss, running_corrects, num_cnt = 0.0, 0, 0

  epoch_t_labels.clear()
  epoch_t_preds.clear()
  
  with torch.no_grad():
    for inputs, labels in dataloaders[phase]:
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)
      num_cnt += len(labels)

      if (phase == 'test'):
        #print('#preds,labels')
        #print(preds.tolist())
        #print(labels.data.tolist())
        epoch_t_preds.extend(preds.tolist())
        epoch_t_labels.extend(labels.data.tolist())
                
  epoch_loss = float(running_loss / num_cnt)
  epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)

  if (phase == 'valid'):            
    valid_loss.append(epoch_loss)
    valid_acc.append(epoch_acc)
  print('#epoch: {} {} loss: {:.2f} acc: {:.1f}'.format(epoch, phase, epoch_loss, epoch_acc))

def print_confusion_matrix():
  #print('#epoch_t_labels, epoch_t_preds')
  #print(epoch_t_preds)
  #print(epoch_t_labels)
  print('#confusion_matrix')
  print(confusion_matrix(epoch_t_labels, epoch_t_preds))
  print('#accuracy: ', accuracy_score(epoch_t_labels, epoch_t_preds))
  print('#precision: ', precision_score(epoch_t_labels, epoch_t_preds))
  print('#recall: ', recall_score(epoch_t_labels, epoch_t_preds))
  print('#f1: ', f1_score(epoch_t_labels, epoch_t_preds))

def plot():
  epochs = []
  for epoch in range(num_epochs):
    epochs.append(epoch)
  #print('#len(epochs), len(train_acc)')
  #print(len(epochs), len(train_acc), len(valid_acc))
  
  #plot epoch vs accuracy
  plt.title('epoch vs accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.plot(epochs, train_acc)
  plt.plot(epochs, train_acc, 'bs', label="train acc")
  plt.plot(epochs, valid_acc)
  plt.plot(epochs, valid_acc, 'rs', label="valid acc")
  plt.legend()
  plt.show()

  #plot epoch vs loss
  plt.title('epoch vs loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.plot(epochs, train_loss)
  plt.plot(epochs, train_loss, 'bs', label="train loss")
  plt.plot(epochs, valid_loss)
  plt.plot(epochs, valid_loss, 'rs', label="valid loss")
  plt.legend()
  plt.show()

#########################################################
### input arguments
num_epochs=10
#num_epochs=100

batch_size  = 64
#batch_size  = 128

model_type = 'efficientnet'
#model_type = 'xception'
print('#model type: ', model_type)

path_kind = 'ds/hq/f2f'
#path_kind = 'ds/hq/nt'
#path_kind = 'ds/lq/f2f'
#path_kind = 'ds/lq/nt'

### make dataset and dataloaders
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

path_train = path_kind+'/train' #ex) ds/hq/f2f/train
path_test = path_kind+'/test' #ex) ds/hq/f2f/test
path_val = path_kind+'/val' #ex) ds/hq/f2f/val 
print("#path_train,test,val: {}, {}, {}".format(path_train, path_test, path_val))

ds_train = datasets.ImageFolder(
  path_train,
  transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))
                                
ds_test = datasets.ImageFolder(
  path_test,
  transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))

ds_val = datasets.ImageFolder(
  path_val,
  transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))

###data loader define
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(ds_train,
  batch_size=batch_size, shuffle=True,
  #!num_workers=4)
  num_workers=0)
dataloaders['valid'] = torch.utils.data.DataLoader(ds_test,
  batch_size=batch_size, shuffle=False,
  num_workers=0)
dataloaders['test']  = torch.utils.data.DataLoader(ds_val,
  batch_size=batch_size, shuffle=False,
  num_workers=0)

class_names = {
  "0": "fake", 
  "1": "real", 
}


### set model and parameters 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2).to(device)
if model_type == 'xception':
  model = Xception().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum=0.9, weight_decay=1e-4)

### run  
for epoch in range(num_epochs):
  train(epoch)
  test(epoch, 'valid')
test(epoch, 'test')
 
print_confusion_matrix()
plot()

###check image data
'''
import torchvision
def imgshow(input_data, title):
  input_data = input_data.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  input_data = std * input_data + mean
  input_data = np.clip(input_data, 0, 1)
  plt.imshow(input_data)
  plt.title(title)
  plt.show()

inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs[0])  
imgshow(out, title="image show")
'''

