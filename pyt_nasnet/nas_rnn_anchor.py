import torch
from torch import sigmoid, tanh, relu_
import torch.nn as nn
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import numpy as np


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


class Reinforce(nn.Module):
    def __init__(self, cell, embedding_dim, num_layers, num_hyper_param, hidden_size, pred_dims):
        super(Reinforce, self).__init__()
        self.emb_dim = embedding_dim
        self.pred_dims = pred_dims  # the length equal to hyper params
        self.num_hp = num_hyper_param
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.reward_buffer = []
        self.state_buffer = []
        self.emb_weights = nn.ParameterList([nn.Parameter(torch.randn(pd, self.emb_dim)) for pd in self.pred_dims])
        self.cells = nn.ModuleList([cell(self.emb_dim, hidden_size), cell(hidden_size, hidden_size)])
        self.predictors = nn.ModuleList([nn.Linear(hidden_size, pred_dim, False) for pred_dim in self.pred_dims])
        self.anchor_v = nn.Parameter(torch.randn(self.pred_dims[-1], 1))

    def get_action(self, state):
        all_hp_probs, all_anchor_probs = self.forward(state)
        all_anchor_act, all_hp_act = [], []
        for layer_anchor_probs in all_anchor_probs:
            anchor_sampler = Bernoulli(layer_anchor_probs)
            layer_anchor_act = anchor_sampler.sample()
            all_anchor_act.append(layer_anchor_act)
        for hp_probs in all_hp_probs:
            sampler = OneHotCategorical(logits=hp_probs)
            all_hp_act.append(sampler.sample())
        return all_hp_act, all_anchor_act

    def store_roll_out(self, state, reward):
        for idx in range(state.size()[0]):
            self.reward_buffer.append(reward[idx])
            self.state_buffer.append(state[idx])

    def forward(self, input_tensor, hidden_states=None):
        all_hp_pred, batch_size, all_anchors = [], input_tensor.size()[0], []
        if hidden_states is None:
            state = tuple([torch.randn(batch_size, self.hidden_size)] * 2)
            hidden_states = [state, state]
        pre_out = input_tensor
        for layer_id in range(self.num_layers):
            for hp_id in range(self.num_hp):
                pre_out = torch.mm(pre_out, self.emb_weights[hp_id])
                for i, cell in enumerate(self.cells):
                    pre_out, out_state = cell(pre_out, hidden_states[i])
                    hidden_states[i] = out_state
                pre_out = self.predictors[hp_id](pre_out)
                if (hp_id % self.num_hp) != self.num_hp - 1:
                    all_hp_pred.append(pre_out)
                else:
                    all_anchors.append(pre_out)
        all_prob_anchors = []
        for idx in range(1, len(all_anchors)):
            layer_anchors = []
            for i in range(idx):
                act_anchor = torch.tanh(torch.add(all_anchors[i], all_anchors[idx]))
                prob_anchor = torch.sigmoid(torch.mm(act_anchor, self.anchor_v))
                layer_anchors.append(prob_anchor)
            layer_anchors = torch.cat(tuple(layer_anchors), dim=-1)
            all_prob_anchors.append(layer_anchors)
        return all_hp_pred, all_prob_anchors


x_net = Reinforce(cell=NASCell,
                  embedding_dim=20,
                  num_layers=7,
                  num_hyper_param=2,
                  hidden_size=35,
                  pred_dims=[5] * 2)
t = F.softmax(torch.randn(3, 5), dim=-1)
x = OneHotCategorical(t).sample()
act = x_net.get_action(x)



