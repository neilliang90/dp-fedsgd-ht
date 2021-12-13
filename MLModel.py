# Several basic machine learning models
import torch
from torch import nn


class LogisticRegression(nn.Module):
    """A simple implementation of Logistic regression model"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)

	
		
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16*5*5, 120)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)     
        #x = x.view(-1, 4*4*50)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        #x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


