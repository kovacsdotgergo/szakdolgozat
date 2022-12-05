import torch
import torch.nn as nn
import collections

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

class Version_enum():
    def __init__(self):
        self.v1 = 0
        self.v2 = 1
        self.v3 = 2
        self.v4 = 3

class Cnn_res_2d(nn.Module):
    def __init__(self, version=Version_enum.v4):
        super(Cnn_res_2d, self).__init__()
        self.version = version

        self.cnns = nn.ModuleList()
        self.cnns[Version_enum.v1] = collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('pool1', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True))#->55*16
        ])     
        self.cnns[Version_enum.v2] = collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('conv2', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('pool3', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True)),#->55*16
            ('conv4', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16)),
            ('conv5', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16)),
            ('conv6', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16))
        ])
        self.cnns[Version_enum.v3] = collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('conv2', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('pool3', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True)),#->55*16
            ('conv4', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16)),
            ('conv5', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16)),
            ('conv6', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16)),
            ('pool6', Res_block(layer_num=2, in_channels=64, out_channels=128,
                in_h=55, in_w=16, pool_first=True)),#->28*8
            ('conv7', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8)),
            ('conv8', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8)),
            ('conv9', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8)),
            ('conv10', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8)),
            ('conv11', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8))
        ])
        self.cnns[Version_enum.v4] = collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('conv2', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('pool3', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True, batch_norm=True)),#->55*16
            ('conv4', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv5', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv6', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('pool6', Res_block(layer_num=2, in_channels=64, out_channels=128,
                in_h=55, in_w=16, pool_first=True, batch_norm=True)),#->28*8
            ('conv7', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv8', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv9', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv10', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv11', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True))
        ])
        self.mlps = nn.ModuleList()
        self.mlps[Version_enum.v1] = torch.nn.Linear(in_features=64*55*16,
                                                     out_features=50)
        self.mlps[Version_enum.v2] = torch.nn.Linear(in_features=64*55*16,
                                                     out_features=50)
        self.mlps[Version_enum.v3] = torch.nn.Linear(in_features=128*28*8,
                                                     out_features=50)
        self.mlps[Version_enum.v4] = torch.nn.Linear(in_features=128*28*8, 
                                                     out_features=50)                                                     
        self.flatten = nn.Flatten(1)
    
    def forward(self, input: torch.tensor) -> torch.tensor:
        #unsqueeze if there is no batch dim
        if 2 == input.dim():
            input = input.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            input = input.unsqueeze(dim=1)
        input = self.cnns[self.version](input)
        input = self.flatten(input)
        input = self.mlps[self.version](input)
        return input
   
class Res_block(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, in_h, in_w,
            pool_first=False, batch_norm=False):
        super(Res_block, self).__init__()
        if pool_first:
            in_h, in_w = div_round_up(in_h, 2), div_round_up(in_w, 2)
        self.layer_num = layer_num
        self.cnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layer_num):
            self.cnns.append(nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if pool_first and i == 0 else 1,
                padding=1))
            if batch_norm:
                self.norms.append(nn.BatchNorm2d(num_features=out_channels))
            else:
                self.norms.append(
                    nn.LayerNorm(normalized_shape=[out_channels, in_h, in_w]))
        if pool_first:
            self.project_residual = nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=2)
        elif in_channels != out_channels:
            self.project_residual = nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1)
        else:
            self.project_residual = None
        self.relu = nn.ReLU()
    
    def forward(self, input:torch.tensor) -> torch.tensor:        
        cnn_input = input.clone()
        for i, (cnn_layer, norm_layer) in enumerate(zip(self.cnns, self.norms)):
            cnn_input = cnn_layer(cnn_input)
            cnn_input = norm_layer(cnn_input)
            if i < self.layer_num - 1:
                cnn_input = self.relu(cnn_input)
        if self.project_residual is not None: 
            input = self.project_residual(input)
        output = cnn_input + input
        output = self.relu(output)
        return output

#source: https://www.folkstalk.com/2022/10/round-up-division-python-with-code-examples.html
def div_round_up(a, b=1):
    return -(-a//b)