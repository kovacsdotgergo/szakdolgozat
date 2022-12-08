import torch
import torch.nn as nn
import collections

#128x512(436) input mel spect
class Conv2d_v1(nn.Module):
    """@brief   class for 2d CNN used on mel spectogram"""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),#128*512
            nn.ReLU(),
            nn.MaxPool2d(2),#64*256
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),#64*256
            nn.ReLU(),
            nn.MaxPool2d(2),#32*128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#32*128
            nn.ReLU(),
            nn.MaxPool2d(2),#16*64
            nn.Dropout(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),#16*64
            nn.ReLU(),
            nn.MaxPool2d(2),#8*32
            nn.Dropout(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),#8*32
            nn.ReLU(),
            nn.MaxPool2d(2),#4*16
            nn.Dropout(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),#4*16
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


class Cnn_res_2d(nn.Module):
    class Version_enum():
        """@brief   enum class for versions of Cnn_res_2d"""
        v1 = 0
        v2 = 1
        v3 = 2
        v4 = 3
        v5 = 4
        v6 = 5
        v7 = 6
        v8 = 7
        v9 = 8

    def __init__(self, version):
        """@brief   constructor
        @param[in]  version     version of the network, indexed by 
                                Version_enum class"""
        super(Cnn_res_2d, self).__init__()
        self.cnns = []
        self.mlps = [] # for storing tuple (in_features, out_features)
        #v1
        self.cnns.append(collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('pool1', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True))#->55*16
        ]))
        self.mlps.append((64*55*16, 50))
        #v2
        self.cnns.append(collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('pool1', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True, batch_norm=True))#->55*16
        ]))
        self.mlps.append((64*55*16, 50))
        #v3
        self.cnns.append(collections.OrderedDict([
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
        ]))
        self.mlps.append((64*55*16, 50))
        #v4
        self.cnns.append(collections.OrderedDict([
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
        ]))
        self.mlps.append((128*28*8, 50))
        #v5
        self.cnns.append(collections.OrderedDict([
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
        ]))
        self.mlps.append((128*28*8, 50))
        #v6
        self.cnns.append(collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv11', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('conv12', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('conv13', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('pool21', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True, batch_norm=True)),#->55*16
            ('conv22', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv23', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv24', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('pool31', Res_block(layer_num=2, in_channels=64, out_channels=128,
                in_h=55, in_w=16, pool_first=True, batch_norm=True)),#->28*8
            ('conv32', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv33', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv34', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv35', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv36', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('pool41', Res_block(layer_num=2, in_channels=128, out_channels=256,
                in_h=28, in_w=8, pool_first=True, batch_norm=True)),#->14*4
            ('conv42', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True)),
            ('conv43', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True))
        ]))
        self.mlps.append((256*14*4, 50))
        #v7
        self.cnns.append(collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv11', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('conv12', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('conv13', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('pool21', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True, batch_norm=True)),#->55*16
            ('conv22', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv23', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv24', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('pool31', Res_block(layer_num=2, in_channels=64, out_channels=128,
                in_h=55, in_w=16, pool_first=True, batch_norm=True)),#->28*8
            ('conv32', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv33', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv34', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv35', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv36', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('pool41', Res_block(layer_num=2, in_channels=128, out_channels=256,
                in_h=28, in_w=8, pool_first=True, batch_norm=True)),#->14*4
            ('conv42', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True)),
            ('conv43', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True)),
            ('conv44', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True)),
            ('pool51', Res_block(layer_num=2, in_channels=256, out_channels=512,
                in_h=14, in_w=4, pool_first=True, batch_norm=True)),#->14*4
            ('conv52', Res_block(layer_num=2, in_channels=512, out_channels=512,
                in_h=7, in_w=2, batch_norm=True)),
            ('conv53', Res_block(layer_num=2, in_channels=512, out_channels=512,
                in_h=7, in_w=2, batch_norm=True))
        ]))
        self.mlps.append((512*7*2, 50))
        #v8
        self.cnns.append(collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv11', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('conv12', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32, batch_norm=True)),
            ('pool21', Res_block(layer_num=2, in_channels=32, out_channels=64,
                in_h=109, in_w=32, pool_first=True, batch_norm=True)),#->55*16
            ('conv22', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('conv23', Res_block(layer_num=2, in_channels=64, out_channels=64,
                in_h=55, in_w=16, batch_norm=True)),
            ('pool31', Res_block(layer_num=2, in_channels=64, out_channels=128,
                in_h=55, in_w=16, pool_first=True, batch_norm=True)),#->28*8
            ('conv32', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('conv33', Res_block(layer_num=2, in_channels=128, out_channels=128,
                in_h=28, in_w=8, batch_norm=True)),
            ('pool41', Res_block(layer_num=2, in_channels=128, out_channels=256,
                in_h=28, in_w=8, pool_first=True, batch_norm=True)),#->14*4
            ('conv42', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True)),
            ('conv43', Res_block(layer_num=2, in_channels=256, out_channels=256,
                in_h=14, in_w=4, batch_norm=True)),
            ('pool51', Res_block(layer_num=2, in_channels=256, out_channels=512,
                in_h=14, in_w=4, pool_first=True, batch_norm=True)),#->14*4
            ('conv52', Res_block(layer_num=2, in_channels=512, out_channels=512,
                in_h=7, in_w=2, batch_norm=True)),
            ('conv53', Res_block(layer_num=2, in_channels=512, out_channels=512,
                in_h=7, in_w=2, batch_norm=True))
        ]))
        self.mlps.append((512*7*2, 50))
        #v9 TODO
        self.cnns.append(collections.OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=16,#436*128->218*64
                kernel_size=5, padding=2, stride=2)),
            ('pool0', nn.Conv2d(in_channels=16, out_channels=32,#->109*32
                kernel_size=3, stride=2, padding=1)),
            ('conv1', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32)),
            ('conv2', Res_block(layer_num=2, in_channels=32, out_channels=32,
                in_h=109, in_w=32))
        ]))
        self.mlps.append((32*109*32, 50))

        # choosing the the correct layers
        self.cnn = nn.Sequential(self.cnns[version])
        in_features, out_features = self.mlps[version]
        self.mlp = nn.Linear(in_features=in_features,
                            out_features=out_features)
        self.flatten = nn.Flatten(1)
    
    def forward(self, input: torch.tensor) -> torch.tensor:
        # unsqueeze if there is no batch dim
        if 2 == input.dim():
            input = input.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            input = input.unsqueeze(dim=1)
        # cnn part of network, depending on the version
        input = self.cnn(input)
        input = self.flatten(input)
        # fully connected output layer
        input = self.mlp(input)
        return input
   
class Res_block(nn.Module):
    """@brief   class for residual cnn block"""
    def __init__(self, layer_num, in_channels, out_channels, in_h, in_w,
            pool_first=False, batch_norm=False):
        """@param[in]   layer_num   number of consecutive cnn layers
        @param[in]      in_channels     number of channels in input
        @param[in]      out_channels    number of channels in output
        @param[in]      in_h        input height
        @param[in]      in_w        input width
        @param[in]      pool_first      if True, first cnn has stride = 2
        @param[in]      batch_norm      uses BatchNorm2d if True,
                                        LayerNorm if False"""
        super(Res_block, self).__init__()
        # halving sizes if first layer has stride = 2
        if pool_first:
            in_h, in_w = div_round_up(in_h, 2), div_round_up(in_w, 2)
        self.layer_num = layer_num
        self.cnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        # stacking the given number of cnns
        for i in range(layer_num):
            # the first layer changes the number of channels 
            # and may have stride = 2 for pooling
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
        # for the residual connection, if the any of the sizes change 
        # then the projection is done with 1x1 conv
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
        # making a copy for the addition later in residual connection
        cnn_input = input.clone()
        for i, (cnn_layer, norm_layer) in enumerate(zip(self.cnns, self.norms)):
            cnn_input = cnn_layer(cnn_input)
            cnn_input = norm_layer(cnn_input)
            # last relu after residual connection
            if i < self.layer_num - 1:
                cnn_input = self.relu(cnn_input)
        if self.project_residual is not None: 
            input = self.project_residual(input)
        # residual connection
        output = cnn_input + input
        output = self.relu(output)
        return output

#source: https://www.folkstalk.com/2022/10/round-up-division-python-with-code-examples.html
def div_round_up(a, b=1):
    return -(-a//b)