import torch
from torch import sigmoid, tanh, relu_
import torch.nn as nn
from torch.distributions.one_hot_categorical import OneHotCategorical


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


class Reinforce(nn.Module):
    def __init__(self, cell, classes, num_cells, hidden_size):
        super(Reinforce, self).__init__()
        self.classes = classes
        self.num_cells = num_cells
        self.hidden_size = hidden_size
        self.reward_buffer = []
        self.state_buffer = []
        self.cells = nn.ModuleList([cell(classes, hidden_size), cell(hidden_size, hidden_size)])
        self.predictors = nn.ModuleList([nn.Linear(hidden_size, self.classes) for _ in range(self.num_cells)])
        self.pred = nn.Softmax(dim=-1)

    def get_action(self, state):
        action_prob = self.forward(state)[0]
        sampler = OneHotCategorical(logits=action_prob)
        return sampler.sample()

    def store_roll_out(self, state, reward):
        for idx in range(state.size()[0]):
            self.reward_buffer.append(reward[idx])
            self.state_buffer.append(state[idx])

    def forward(self, input_tensor, hidden_states=None):
        outputs, batch_size = [], input_tensor.size()[0]
        if hidden_states is None:
            state = tuple([torch.randn(batch_size, self.hidden_size)] * 2)
            hidden_states = [state, state]
        pre_out = input_tensor
        for time in range(self.num_cells):
            for i, cell in enumerate(self.cells):
                pre_out, out_state = cell(pre_out, hidden_states[i])
                hidden_states[i] = out_state
            pre_out = self.predictors[time](pre_out)
            outputs.append(pre_out)
        # get softmax prob
        return self.pred(torch.stack(tuple(outputs), dim=1)), hidden_states

