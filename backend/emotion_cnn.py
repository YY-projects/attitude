import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # -> (32, 256, 256)
        self.pool1 = nn.MaxPool2d(2, 2)                          # -> (32, 128, 128)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64, 128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)                          # -> (64, 64, 64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> (128, 64, 64)
        self.pool3 = nn.MaxPool2d(2, 2)                           # -> (128, 32, 32)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x