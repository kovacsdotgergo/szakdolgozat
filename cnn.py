import torch
import torch.nn as nn

#128x512(436) input mel spect
class conv2d_v1(nn.Module):
    """@brief   class for 2d CNN used on mel spectogram"""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#128*512
            nn.ReLU(),
            nn.MaxPool2d(2),#64*256
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#64*256
            nn.ReLU(),
            nn.MaxPool2d(2),#32*128
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#32*128
            nn.ReLU(),
            nn.MaxPool2d(2),#16*64
            nn.Dropout(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#16*64
            nn.ReLU(),
            nn.MaxPool2d(2),#8*32
            nn.Dropout(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#8*32
            nn.ReLU(),
            nn.MaxPool2d(2),#4*16
            nn.Dropout(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#4*16
            nn.ReLU(),
            nn.MaxPool2d(2)#2*8
        )
        #flatten all dimensions except batch
        self.flatten = nn.Flatten(1)
        #fully connected layer
        self.mlp = nn.Sequential(
            nn.Linear(2*8*512, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 50),
        )
    
    def forward(self, x):
        #passing through all layers
        x = torch.unsqueeze(x, 1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.mlp(x)

class cnn2d_v2(nn.Module):
    """TODO class for residual deep cnn with batchnorm"""
    def __init__(self):
        raise NotImplementedError