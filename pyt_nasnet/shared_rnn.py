import torch
import torch.nn.functional as F
from torch import nn
from torch import sigmoid, tanh, relu_


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        self.dropout = dropout
        super().__init__()

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = m / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout=0.0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def set_weights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1)
                if raw_w.is_cuda: mask = mask.cuda()
                mask = F.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)
            w = nn.Parameter(w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self.set_weights()
        return self.module.forward(*args)


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


class R2N2_VAR(nn.Module):
    def __init__(self, num_inputs, rnn_config):
        super(R2N2_VAR, self).__init__()
        num_hidden, P, num_layer, dropout = rnn_config
        self.encoder_rnn = nn.LSTM(num_inputs, num_hidden, 2, batch_first=True, bidirectional=True)
        self.encoder_rnn = nn.GRUCell
        self.wdec = WeightDrop(self.encoder_rnn, ['weight_hh_l0', 'weight_ih_l0'], dropout=dropout)
        self.A = nn.ModuleList([nn.Linear(num_inputs, num_inputs) for _ in range(self.P)])
        self.lock_drop = LockedDropout(dropout)
        self.output_layer = nn.Linear(num_hidden * 2, num_inputs)

    def forward(self, y, last_state=None):
        sum_wyb = torch.zeros_like(y[:, 0, :])
        for idx, layer in enumerate(self.A):
            sum_wyb += layer(y[:, idx, :])
        encoded_y, hidden_state = self.wdec(y, last_state)
        encoded_y = self.lock_drop(encoded_y)
        outputs = self.output_layer(encoded_y[:, -1, :])
        return outputs + sum_wyb
