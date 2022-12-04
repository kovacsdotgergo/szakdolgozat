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

class Cnn2d_v2(nn.Module):
    """TODO class for residual deep cnn with batchnorm"""
    def __init__(self):
        super(cnn2d_v2, self).__init__()
        self.cnn1 = Cnn_block(layer_num=3, in_channels=1, out_channels=16,
            in_h=436, in_w=128)
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.mlp = torch.nn.Linear(in_features=16*218*64, out_features=50)
    
    def forward(self, input: torch.tensor) -> torch.tensor:
        #unsqueeze if there is no batch dim
        if 2 == input.dim():
            input = input.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            input = input.unsqueeze(dim=1)
        input = self.cnn1(input)
        input = self.pool1(input)
        input = self.flatten(input)
        input = self.mlp(input)
        return input

class Cnn_block(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, in_h, in_w):
        super(Cnn_block, self).__init__()
        self.layer_num = layer_num
        self.cnns = []
        self.layer_norms = []
        for i in range(layer_num):
            self.cnns.append(nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)))
            self.layer_norms.append(
                nn.LayerNorm(normalized_shape=[out_channels, in_h, in_w]))
            
        if in_channels != out_channels:
            self.project_residual = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(1, 1),
                                          stride=(1, 1))
        else:
            self.project_residual = None
        self.relu = nn.ReLU()
    
    def forward(self, input:torch.tensor) -> torch.tensor:        
        cnn_input = input.clone()
        for i, (cnn_layer, norm_layer) in enumerate(zip(self.cnns, self.layer_norms)):
            cnn_input = cnn_layer(cnn_input)
            cnn_input = norm_layer(cnn_input)
            if i < self.layer_num - 1:
                cnn_input = self.relu(cnn_input)
        if self.project_residual is not None: 
            input = self.project_residual(input)
        output = cnn_input + input
        output = self.relu(output)
        return output