from torch import nn
import torch # Added import for torch.flatten

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Added pooling layer

        # Changed out_channels to 256 to match the input of fc1 after adaptive pooling
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Added pooling layer

        # Added adaptive pooling to get a 1x1 spatial output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool1(x)        # B x 64 x 112 x 112

        x = self.conv2(x).relu() # B x 256 x 112 x 112 (after conv2 change)
        x = self.pool2(x)        # B x 256 x 56 x 56

        x = self.adaptive_pool(x) # B x 256 x 1 x 1
        x = torch.flatten(x, 1)   # B x 256 (flatten from dimension 1)

        x = self.fc1(x)
        return x