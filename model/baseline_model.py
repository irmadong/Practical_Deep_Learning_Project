import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    """
    Baseline CNN model
    credit to: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BaseNet_test(nn.Module):
    """
    Baseline CNN model
    credit to: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BaseNet_100(nn.Module):
    """
    Baseline CNN model
    credit to: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool(self.conv2(F.relu(self.conv1(x))))
        x = self.pool(self.conv4(F.relu(self.conv3(x))))
        x = self.pool(self.conv6(F.relu(self.conv5(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# +
# class BaseNet_100(nn.Module):
#     """
#     Baseline CNN model
#     credit to: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#     """

#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)
#         self.pool = nn.MaxPool2d(2, 2) 
#         self.fc1 = nn.Linear(256 * 4 * 4, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 100)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = self.pool(self.conv6(x))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x

# +
# class Cifar100CnnModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

#             nn.Flatten(), 
#             nn.Linear(256*4*4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 100))
        
#     def forward(self, xb):
#         return self.network(xb)
