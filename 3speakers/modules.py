import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch
from itertools import permutations


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='LSTM', bidirectional = True, batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)] # 2^(n_layers-1)
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout, bidirectional = bidirectional)
            else:
                if bidirectional:
                    c = cell(n_hidden*2, n_hidden, dropout=dropout, bidirectional = bidirectional)
                else:
                    c = cell(n_hidden, n_hidden, dropout=dropout, bidirectional = bidirectional)
            layers.append(c)
        self.cells = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs = self.dropout(inputs)
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None): # rate means dilation
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate) # does nothing on even inputs
        dilated_inputs = self._prepare_inputs(inputs, rate) # different phases concatenated in batch dimension

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None: # init to zeros
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                if self.bidirectional:
                    hidden = (c.unsqueeze(0).repeat(2, 1, 1), m.unsqueeze(0).repeat(2, 1, 1))
                else:
                    hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        # dilated_outputs.shape = L, batch*rate, n_hidden
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)] # [rate, L, batch, n_hidden]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous() # [L, rate, batch, n_hidden]
        interleaved = interleaved.view(dilated_outputs.size(0) * rate, 
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda(device)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim).cuda(device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim).cuda(device)
            return (hidden, memory)
        else:
            return hidden
        

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 3, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class AttentionModule_stage1(nn.Module):
    # input size is 56*56
    def __init__(self, in_channels, out_channels, size1=(128, 128), size2=(64, 64), size3=(32, 32), out_skip = True):
        super(AttentionModule_stage1, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, in_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, in_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, in_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = ResidualBlock(in_channels, in_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, in_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels)
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = ResidualBlock(in_channels, in_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = ResidualBlock(in_channels, in_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)
        self.out_skip = out_skip
        
    def forward(self, x):
        #batch_size, nheads, length, n_mels = x.shape
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x) # 100x64
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1) # 50x32
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2) # 25x16
        out_softmax3 = self.softmax3_blocks(out_mpool3) 
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5) + out_trunk
        out_softmax6 = self.softmax6_blocks(out_interp1)
        if self.out_skip:
            out = (1 + out_softmax6) * out_trunk
        else:
            out = out_softmax6 * out_trunk
        out_last = self.last_blocks(out)
        return out_last

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, dilation = dilation, padding=padding, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = nn.BatchNorm2d(in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)

class ResDilationBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResDilationBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels, 3, padding = 1, bias = False, dilation = 1)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(input_channels, input_channels, 3, padding = 2, bias = False, dilation = 2)
        self.bn3 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(input_channels, input_channels, 3, padding = 4, bias = False, dilation = 4)
        self.bn4 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(input_channels, input_channels, 3, padding = 8, bias = False, dilation = 8)
        self.bn5 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(input_channels, input_channels, 3, padding = 16, bias = False, dilation = 16)
        self.bn6 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(input_channels, input_channels, 3, padding = 32, bias = False, dilation = 32)
        self.bn7 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(input_channels, output_channels, 1, bias = False)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        residual = out
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn6(out)
        out = self.relu(out)
        out = self.conv6(out)
        out += residual
        out = self.bn7(out)
        out = self.relu(out)
        out = self.conv7(out)
        
        return out

def cal_loss(source, estimate_source):
    """
    Args:
        source: [B, C, T, F], B is batch size, F is frequency
        estimate_source: [B, C, T, F]
        source_lengths: [B, F]
    """
    min_mse, perms, min_mse_idx = cal_mse(source,estimate_source)
    loss = torch.mean(min_mse)
    reorder_estimate_source = reorder_source(estimate_source, perms, min_mse_idx)
    return loss, min_mse, estimate_source, reorder_estimate_source


def cal_mse(source, estimate_source):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T, F], B is batch size
        estimate_source: [B, C, T, F]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T, F = source.size()


    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(source, dim=1)  # [B, 1, C, T, F]
    s_estimate = torch.unsqueeze(estimate_source, dim=2)  # [B, C, 1, T, F]
    # s_target = <s', s>s / ||s||^2
    pair_wise_mse = torch.sum((s_estimate - s_target)**2, dim=(3, 4))/T/F  # [B, C, C]

    ############## since distribution is asymmetric, use l2 not l1 loss ###########
    
    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    mse_set = torch.einsum('bij,pij->bp', [pair_wise_mse, perms_one_hot])
    min_mse_idx = torch.argmin(mse_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    min_mse, _ = torch.min(mse_set, dim=1, keepdim=True)
    min_mse /= C
    return min_mse, perms, min_mse_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source
