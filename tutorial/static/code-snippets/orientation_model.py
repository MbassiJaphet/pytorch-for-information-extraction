import torch.nn as nn
import torch.nn.functional as F

class OrientationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(OrientationModel, self).__init__()
        # Define convolutional layers
        self.layer_1_conv = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)  # W = 112
        self.layer_2_conv = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)  # W = 56
        self.layer_3_conv = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)  # W = 28

        # Define maxpool and dropout layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)

        # Define the fully-connected or linear layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 512)
        # The outputs of the last linear equals the number of orientation classes
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.layer_1_conv(x)))
        x = self.pool(F.relu(self.layer_2_conv(x)))
        x = self.pool(F.relu(self.layer_3_conv(x)))

        x = x.view(-1, 128 * 28 * 28)
        # Insert an activation layer after every linear layer
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))

        return x # return orientation prediction