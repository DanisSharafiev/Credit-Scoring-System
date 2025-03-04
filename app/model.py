import torch
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(53, 128)
        self.dropout1 = nn.Dropout(0.3)  

        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.4)  

        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.2)  

        self.fc5 = nn.Linear(32, 32)
        self.dropout5 = nn.Dropout(0.1)

        self.fc6 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)
    
        x = self.relu(self.fc5(x))
        x = self.dropout5(x)

        x = self.fc6(x)
        return x