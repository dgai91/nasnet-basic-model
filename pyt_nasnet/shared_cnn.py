from torch import nn
import numpy as np


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


def gen_fc_dim(cnn_config, feathers):
    for idd, filter_size in enumerate(cnn_config[0]):
        feathers = (feathers - int(filter_size) + 1 - int(cnn_config[2][idd]) + 1) / 2
        feathers = int(np.ceil(feathers))
    print(feathers)
    return feathers


class CNN(nn.Module):

    def __init__(self, num_input, ic, num_class, cnn_config):
        super(CNN, self).__init__()
        cnn_ksize, num_filters, max_pool_ksize = cnn_config
        self.cnn_config = cnn_config
        block_list = []
        for idd, filter_size in enumerate(cnn_ksize):
            block = nn.Sequential()
            block.add_module('conv_out_' + str(idd), nn.Conv2d(ic, num_filters[idd], int(filter_size), 1))
            block.add_module('relu_' + str(idd), nn.ReLU())
            block.add_module('max_pool_' + str(idd), nn.MaxPool2d(int(max_pool_ksize[idd]), 2))
            block.add_module('dropout_' + str(idd), nn.Dropout(0.5))
            block_list.append(block)
            ic = num_filters[idd]

        self.flatten = Flatten()
        feathers = gen_fc_dim(cnn_config, num_input)
        self.logits = nn.Linear(ic * feathers * feathers, num_class)
        self.block_list = nn.ModuleList(block_list)

    def forward(self, x, **kwargs):
        for block in self.block_list:
            x = block(x)
        x = self.flatten(x)
        out = self.logits(x)
        return out
