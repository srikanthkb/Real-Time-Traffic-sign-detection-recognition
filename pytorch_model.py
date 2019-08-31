import torch 
import numpy as np 
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,1,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,1,padding=1)
        self.fc1 = nn.Linear(12*12*64,600)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(600,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 12*12*64)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    def im_convert(tensor):
        image = tensor.cpu().clone().detach().numpy()
        image = image.transpose(1,2,0)
        image = image*np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))
        image = image.clip(0,1)
        return image

 



classes = ('Left Turn','No Entry','No Horn','No Left Turn','No Stoppping','No U-Turn','Pedestrian Crossing','Right Turn','Speed Breaker','Junction Ahead')

