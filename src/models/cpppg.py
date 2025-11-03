import torch
from torch import nn
import sys
sys.path.append(".")
from src.models.model_utils import *
from src.utils.utils import get_config
from configs.seed import *


class Generator(nn.Module):
    def __init__(self, cfgs):
        super(Generator, self).__init__()
        self.cfgs = cfgs
        self.depth = self.cfgs['model']['baseline']['depth']
        self.use_se_block = self.cfgs['model']['baseline']['use_se_block']
        self.initial_features = self.cfgs['model']['baseline']['initial_features']
        self.features = [self.initial_features * (2**i) for i in range(self.depth)]    
        self.seq_lengths = [self.cfgs['data']['segment_length'] // (2**i) for i in range(self.depth)]    
        self.img_ch = self.cfgs['model']['baseline']['chin']
        self.output_ch = self.cfgs['model']['baseline']['chout']
        self.use_lstm = self.cfgs['model']['baseline']['use_lstm']
        self.use_gated_block = self.cfgs['model']['baseline']['use_gated_block']
        self.gn_in = self.cfgs['data']['gn_input']
        
        if self.gn_in:
            self.first_gn = nn.GroupNorm(1, 1)
            self.film_gn = nn.GroupNorm(1, 1)
            self.last_gn = nn.GroupNorm(1, 1)

        # Encoding path
        self.encoders = nn.ModuleList()
        self.encoder_films = nn.ModuleList()

        self.encoders.append(Conv_block(self.img_ch, self.features[0], use_gated_block=self.use_gated_block, use_se_block=self.use_se_block))

        for i in range(1, self.depth):
            self.encoders.append(Conv_block(self.features[i-1], self.features[i], use_gated_block=self.use_gated_block, use_se_block=self.use_se_block))

        # lstm layer
        if self.use_lstm:
            self.lstm = BLSTM(self.features[-1], bi = True)

        # Decoding path
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(self.depth-1, -1, -1):
            self.up_convs.append(Up_conv(self.features[i], self.features[i-1] if i != 0 else self.features[0], use_gated_block=self.use_gated_block))
            in_channels = self.features[i] + (self.features[i-1] if i != 0 else self.features[0])
            out_channels = self.features[i-1] if i != 0 else self.initial_features  
            self.decoders.append(Conv_block(in_channels, out_channels, use_gated_block=self.use_gated_block, use_se_block=self.use_se_block))

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Dropout = nn.Dropout(p=0.2)
        self.Conv_1x1 = nn.Conv1d(self.features[0], self.output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, seg_film = None, mask_tensor=None):
        skip_connections = []
        if self.gn_in:
            x = self.first_gn(x)
        
        # encoding path
        for i in range(self.depth):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.Maxpool(x)
            x = self.Dropout(x)
            
        # lstm layer
        if self.use_lstm:
            x = x.permute(2, 0, 1)
            x, _ = self.lstm(x)
            x = x.permute(1, 2, 0)

        # decoding path
        for i in range(self.depth):
            x = self.up_convs[i](x)  
            x = torch.cat((skip_connections[-(i+1)], x), dim=1)
            x = self.Dropout(x)
            x = self.decoders[i](x)  

        x = self.Conv_1x1(x)
        if self.gn_in:
            x = self.last_gn(x)
            
        return x


class Discriminator(nn.Module):
    def __init__(self, cfgs):
        super(Discriminator, self).__init__()
        self.cfgs = cfgs
        self.depth = self.cfgs['model']['baseline']['depth']
        self.use_se_block = self.cfgs['model']['baseline']['use_se_block']
        self.initial_features = self.cfgs['model']['baseline']['initial_features']
        self.features = [self.initial_features * (2**i) for i in range(self.depth)]    
        self.seq_lengths = [self.cfgs['data']['segment_length'] // (2**i) for i in range(self.depth)]    
        self.img_ch = self.cfgs['model']['baseline']['chin']
        self.output_ch = self.cfgs['model']['baseline']['chout']
        self.use_lstm = self.cfgs['model']['baseline']['use_lstm']
        self.use_gated_block = self.cfgs['model']['baseline']['use_gated_block']
        self.gn_input = self.cfgs['data']['gn_input']
        self.use_hinge = self.cfgs['train']['hinge']

        self.encoders = nn.ModuleList()

        if self.gn_input:
            self.first_gn = nn.GroupNorm(1, 1)

        self.encoders.append(Conv_block(self.img_ch, self.features[0], use_gated_block=self.use_gated_block, use_se_block=self.use_se_block))
        
        for i in range(1, self.depth):
            self.encoders.append(Conv_block(self.features[i-1], self.features[i], use_gated_block=self.use_gated_block, use_se_block=self.use_se_block))

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Dropout = nn.Dropout(p=0.2)

        if self.use_hinge:
            self.adv_layer = nn.Sequential(nn.Linear(128 * 128, 1))
        else:
            self.adv_layer = nn.Sequential(nn.Linear(128 * 128, 1), nn.Sigmoid())

    def forward(self, x):
        if self.gn_input:
            x = self.first_gn(x)
            
        for i in range(self.depth):
            x = self.encoders[i](x)
            x = self.Maxpool(x)
            x = self.Dropout(x)

        out = x.view(x.size(0), -1)
        out = self.adv_layer(out)
        
        return out


if __name__ == "__main__":
    cfgs = get_config()
    model = Generator(cfgs)
    x = torch.randn(1, 1, 128 * 8)
    out = model(x, x)
    print(out.shape)
    
    model = Discriminator(cfgs)
    x = torch.randn(1, 1, 128 * 8)
    y = model(x)
    print(y.shape)