import torch 
from torch import nn
from torch.nn import functional as F


class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, use_gated_block=True, use_se_block=True, num_groups=16):
        super(Conv_block, self).__init__()
        if use_gated_block:
            conv1d = GatedConv1d
        else:
            conv1d = nn.Conv1d
        self.conv = nn.Sequential(
            conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, ch_out),
            nn.ReLU(inplace=True),
            conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, ch_out),
            nn.ReLU(inplace=True)
        )
        self.use_se_block = use_se_block
        if use_se_block:
            self.se = SEModule(ch_out)

    def forward(self, x):
        x = self.conv(x)
        if self.use_se_block:
            x = self.se(x)
        return x


class Up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, use_gated_block=False, num_groups=16):
        super(Up_conv, self).__init__()
        if use_gated_block:
            conv1d = GatedConv1d
        else:
            conv1d = nn.Conv1d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    

class GatedConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(GatedConv1d, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate_conv = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_out = self.conv(x)
        gate = self.sigmoid(self.gate_conv(x))
        return conv_out * gate
    

class SEModule(nn.Module):
    def __init__(self, channels=1):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels//2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels//2, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x
    

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden
    

class Conv1dSamePadding(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        padding = (
            self.stride[0] * (inputs.shape[-1] - 1)
            - inputs.shape[-1]
            + self.kernel_size[0]
            + (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
        ) // 2
        return self._conv_forward(
            F.pad(inputs, (padding, padding)),
            self.weight,
            self.bias,
        )


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSamePadding(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            Conv1dSamePadding(
                in_channels, out_channels, kernel_size=1, device=device, dtype=dtype
            ),
        )

    def forward(self, inputs):
        return self.conv(inputs)
    

class DeformableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(DeformableConv1d, self).__init__()
        self.offset_conv = nn.Conv1d(in_channels,  in_channels * kernel_size, kernel_size, padding=padding, stride=stride)
        self.regular_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        offsets = self.offset_conv(x)
        B, C, L = x.size()
        indices = torch.arange(L).float().unsqueeze(0).unsqueeze(0).repeat(B, 1, 1).to(x.device)
        indices = indices + offsets
        indices = torch.clamp(indices, 0, L-1)
        indices_floor, indices_ceil = torch.floor(indices), torch.ceil(indices)
        x_floor, x_ceil = x.gather(2, indices_floor.long()), x.gather(2, indices_ceil.long())
        x_deform = (1.0 - (indices - indices_floor)) * x_floor + (indices - indices_floor) * x_ceil
        out = self.regular_conv(x_deform)

        return out


class FiLMBlock(nn.Module):
    def __init__(self, hidden=None, input_len=None):
        super(FiLMBlock, self).__init__()
        self.gamma_generator = nn.Linear(input_len, hidden)
        self.beta_generator = nn.Linear(input_len, hidden)
        
    def forward(self, x, supported_ref_bank):
        supported_ref_bank = supported_ref_bank.reshape(x.size(0), 1, -1)
        
        gamma = self.gamma_generator(supported_ref_bank)
        beta = self.beta_generator(supported_ref_bank)

        gamma = gamma.view(x.size(0), x.size(1), -1)
        beta = beta.view(x.size(0), x.size(1), -1)
        
        x = x * gamma + beta
        return x
