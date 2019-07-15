import torch
from torch import sigmoid, tanh, relu_
import torch.nn as nn


class NASCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_proj=None, use_biases=False):
        super(NASCell, self).__init__()
        self._num_units = hidden_size
        self._num_proj = num_proj

        self._use_biases = use_biases
        self._input_size = input_size

        num_proj = self._num_units if num_proj is None else num_proj
        self.concat_w_m = nn.Parameter(torch.randn(num_proj, 8 * self._num_units))
        self.concat_w_inputs = nn.Parameter(torch.randn(self._input_size, 8 * self._num_units))
        if use_biases:
            self.bias = nn.Parameter(torch.randn(8 * self._num_units))
        if self._num_proj is not None:
            self.concat_w_proj = nn.Parameter(torch.randn(self._num_units, 8 * self._num_proj))

    def forward(self, input, state):
        (m_prev, c_prev) = state
        m_matrix = torch.mm(m_prev, self.concat_w_m)
        input_matrix = torch.mm(input, self.concat_w_inputs)
        if self._use_biases:
            m_matrix = torch.add(m_matrix, self.b)

        m_matrix_splits = torch.split(m_matrix, self._num_units, dim=1)
        inputs_matrix_splits = torch.split(input_matrix, self._num_units, dim=1)

        layer1_0 = sigmoid(inputs_matrix_splits[0] + m_matrix_splits[0])
        layer1_1 = relu_(inputs_matrix_splits[1] + m_matrix_splits[1])
        layer1_2 = sigmoid(inputs_matrix_splits[2] + m_matrix_splits[2])
        layer1_3 = relu_(inputs_matrix_splits[3] * m_matrix_splits[3])
        layer1_4 = tanh(inputs_matrix_splits[4] + m_matrix_splits[4])
        layer1_5 = sigmoid(inputs_matrix_splits[5] + m_matrix_splits[5])
        layer1_6 = tanh(inputs_matrix_splits[6] + m_matrix_splits[6])
        layer1_7 = sigmoid(inputs_matrix_splits[7] + m_matrix_splits[7])

        l2_0 = tanh(layer1_0 * layer1_1)
        l2_1 = tanh(layer1_2 + layer1_3)
        l2_2 = tanh(layer1_4 * layer1_5)
        l2_3 = sigmoid(layer1_6 + layer1_7)

        l2_0 = tanh(l2_0 + c_prev)

        l3_0_pre = l2_0 * l2_1

        new_c = l3_0_pre
        l3_0 = l3_0_pre

        l3_1 = tanh(l2_2 + l2_3)
        new_m = tanh(l3_0 * l3_1)
        if self._num_proj is not None:
            new_m = torch.mm(new_m, self.concat_w_proj)
        return new_m, (new_m, new_c)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state):
        inputs = input.unbind(1)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs, dim=1), state

