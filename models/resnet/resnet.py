''' 
  Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
'''

import torch  
import torch.nn as nn  
import torch.nn.functional as F   

''' 
TODO : use nn.Sequential()  
'''  
class BasicBlock(nn.Module):  
  expansion = 1 
  def __init__(self,in_channels,out_channels,downsample,stride):  
    super().__init__()   
    '''
    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size = 3, padding = 1 , stride = stride , bias = False) 
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size = 3 , padding = 1 , stride = stride , bias = False) 
    self.bn2 = nn.BatchNorm2d(out_channels) 
    self.downsample = downsample
    self.stride = stride 
    self.relu = nn.ReLU()  
    '''  
    self.residual_function = nn.Sequential(
      nn.Conv2d(in_channels,out_channels,kernel_size = 3, padding = 1 , stride = stride , bias = False),
      nn.BatchNorm2d(out_channels), 
      nn.ReLU(inplace = True), # direct 
      nn.Conv2d(out_channels,out_channels,kernel_size = 3 , padding = 1 , stride = stride , bias = False) ,
      nn.BatchNorm2d(out_channels) 
    ) 
    self.shortcut = nn.Sequential()
    if stride!=1 or in_channels!=out_channels: 
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = stride , bias = False),
        nn.BatchNorm2d(out_channels)
      )
  
  def forward(self,x):   
    return nn.ReLU(inplace = True)(self.residual_function(x)+self.shortcut(x))   
    '''
    identity = x.clone() 
    x = self.relu(self.bn2(self.conv1(x)))
    x = self.bn2(self.conv2(x)) 
    if self.downsample is not None : 
      identity = self.downsample(identity) 
    x+=identity() 
    x = self.relu(x) 
    return x 
    '''
class Bottleneck(nn.Module):  
  expansion = 4 
  def __init__(self,in_channels,out_channels,downsample = None,stride = 1):    
    super().__init__()
    '''
    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size = 1 , stride = 1 ,padding = 0)
    self.bn1 = nn.BatchNorm2d(out_channels) 
 
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size = 3 , stride = stride ,padding = 1)
    self.bn2 = nn.BatchNorm2d(out_channels) 
    self.conv3 = nn.Conv2d(out_channels,out_channels*Bottleneck.expansion,kernel_size = 1 , stride = 1 ,padding = 0)
    self.bn3 = nn.BatchNorm2d(out_channels*Bottleneck.expansion) 
    self.downsample = downsample
    self.relu = nn.ReLU()  
    self.stride = stride
    ''' 
    self.residual_function = nn.Sequential(
      nn.Conv2d(in_channels,out_channels,kernel_size = 1 , stride = 1 ,padding = 0),
      nn.BatchNorm2d(out_channels),  
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels,out_channels,kernel_size = 3 , stride = stride ,padding = 1),
      nn.BatchNorm2d(out_channels) ,
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels,out_channels*Bottleneck.expansion,kernel_size = 1 , stride = 1 ,padding = 0),
      nn.BatchNorm2d(out_channels*Bottleneck.expansion) 
 
    )
    self.shortcut = nn.Sequential()
    if stride!=1 or in_channels!=out_channels*Bottleneck.expansion: 
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size = 1 , stride = stride , bias = False),
        nn.BatchNorm2d(out_channels)
      )
  def forward(self,x):  
    return nn.ReLU(inplace=True)(self.residual_function(x)+self.shortcut(x))
    '''
     identity  = x.clone() 
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x))) 
    x = self.conv3(x) 
    if self.downsample is not None :  
      identity = self.downsample(identity) 
    x+=identity 
    x = self.relu(x)
    return x  
  '''
class ResNet(nn.Module): 
  def __init__(self,ResBlock,layer_list,num_classes,num_channels = 3):  
    super().__init__() 
    self.in_channels = 64 
    self.conv1 = nn.Conv2d(num_channels,64,kernel_size = 7 , stride = 2 , padding = 3, bias = False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()  
    self.max_pool = nn.MaxPool2d(kernel_size = 3,stride = 2, padding = 1)  
    self._conv1 = nn.Sequential(
      nn.Conv2d(3,64,kernel_size = 3, padding = 1 , bias = False), 
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True) 
    )
    self.layer1 = self._make_layer(ResBlock,layer_list[0],planes = 64)
    self.layer2 = self._make_layer(ResBlock,layer_list[1],planes = 128 , stride = 2)
    self.layer3 = self._make_layer(ResBlock,layer_list[2],planes = 256 , stride = 2)
    self.layer4 = self._make_layer(ResBlock,layer_list[3],planes = 512 , stride = 2) 
    self.avgpool = nn.AdaptiveAvgPool2d((1,1)) 
    self.fc = nn.Linear(512*ResBlock.expansion,num_classes) 
  def forward(self,x):    
    x = self.relu(self.bn1(self.conv1(x))) 
    x = self.max_pool(x) 
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x) 
    x = self.avgpool(x)  
    x.reshape(x.shape[0],-1)
    x = self.fc(x)
    return x  
  
  def _make_layer(self,ResBlock,blocks,planes,stride=1):  
    i_downsample = None
    layers = []  
    if stride != 1 or self.in_channels != planes*ResBlock.expansion : 
      i_downsample = nn.Sequential(
        nn.Conv2d(self.in_channels,planes * ResBlock.exapnsion,kernel_size = 1,stride = stride),
        nn.BatchNorm2d(planes*ResBlock.expansion) 
      ) 
    layers.append(ResBlock(self.in_channels,planes,downsample = i_downsample,stride = stride))   
    self.in_channels = planes * ResBlock.epansion
    for _ in range(blocks - 1):
      layers.append(ResBlock(self.in_channels,planes))  

    return nn.Sequential(*layers)
  
  def ResNet18(num_classes ,channels = 3): 
    return ResNet(BasicBlock,[2,2,2,2],num_classes, channels)
  def ResNet34(num_classes ,channels = 3): 
    return ResNet(BasicBlock,[3,4,6,3],num_classes, channels)
  def ResNet50(num_classes ,channels = 3): 
    return ResNet(Bottleneck,[3,4,6,3],num_classes, channels)
  def ResNet101(num_classes ,channels = 3): 
    return ResNet(Bottleneck,[3,4,23,3],num_classes, channels)
  def ResNet152(num_classes ,channels = 3):
    return ResNet(Bottleneck,[3,8,36,3],num_classes, channels)