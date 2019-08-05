from torch import nn
import torch
import numpy as np


# TODO: layer connection
class DenseModel(nn.Module):

    def __init__(self, num_input, num_classes, cnn_config):
        super(DenseModel, self).__init__()
        hidden_units, self.all_layer_anchors = cnn_config
        self.all_layers = []
        pre_out_unit = 0
        self.all_layers.append(nn.Linear(num_input, hidden_units[0]))
        for idd, layer_anchor in enumerate(self.all_layer_anchors):
            if sum(layer_anchor) != 0:
                for idx, anchor in enumerate(layer_anchor):
                    if anchor == 1:
                        pre_out_unit += hidden_units[idx]
            else:
                pre_out_unit = num_input
            self.all_layers.append(nn.Linear(pre_out_unit, hidden_units[idd + 1]))
            pre_out_unit = 0
        self.all_layers.append(nn.Linear(hidden_units[-1], num_classes))
        self.all_layers = nn.ModuleList(self.all_layers)

    def forward(self, x, **kwargs):
        hidden = self.all_layers[0](x)
        all_output = [hidden]
        for idx, layer in enumerate(self.all_layers[1:-1]):
            layer_anchor = self.all_layer_anchors[idx]
            if sum(layer_anchor) == 0:
                pre_out = [x]
            else:
                pre_out = []
                for idd, anchor in enumerate(layer_anchor):
                    if anchor == 1:
                        pre_out.append(all_output[idd])
            pre_out = torch.cat(tuple(pre_out), dim=-1)
            pre_out = layer(pre_out)
            all_output.append(pre_out)
        output = self.all_layers[-1](all_output[-1])
        return output
