import torch
import random
import numpy as np
from torch import nn
from pyt_nasnet.nn_nas_rnn import LSTMLayer, NASCell


class Reinforce(nn.Module):
    def __init__(self, max_layers, input_size,
                 division_rate=100.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=1):
        super(Reinforce, self).__init__()
        self.input_size = input_size
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.max_layers = max_layers
        self.exploration = exploration
        self.reward_buffer = []
        self.state_buffer = []
        self.policy_network = LSTMLayer(NASCell, self.input_size, 4 * self.max_layers)
        self.bias = nn.Parameter(torch.FloatTensor([0.05] * 4 * self.max_layers))

    def init_hidden(self):
        state = (torch.randn(1, 4 * self.max_layers), torch.randn(1, 4 * self.max_layers))
        return state

    def get_action(self, state):
        if random.random() < self.exploration:
            action = np.array([[random.sample(range(1, 35), 4 * self.max_layers)]])
            return torch.from_numpy(action)
        else:
            hidden = self.init_hidden()
            state = state.unsqueeze(-1)
            action_scores, state = self.policy_network(state, hidden)
            return torch.ceil(self.division_rate * action_scores[:, -1, :].unsqueeze(1))

    def store_roll_out(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def forward(self, state):
        hidden = self.init_hidden()
        state = state.unsqueeze(-1)
        log_prob, state = self.policy_network(state, hidden)
        return log_prob
