import torch
import random
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
        self.policy_layer = LSTMLayer(NASCell, self.input_size, 100)
        self.output_layer = LSTMLayer(NASCell, 100, self.input_size)

    def init_hidden_state(self, hidden_size):
        state = (torch.randn(1, hidden_size), torch.randn(1, hidden_size))
        return state

    def get_action(self, state):
        if random.random() < self.exploration:
            rand_action = torch.randn_like(state)
            return rand_action
        else:
            self.forward(state)

    def store_roll_out(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def forward(self, state):
        hidden_state = self.init_hidden_state(100)
        hidden, hidden_state = self.policy_layer(state, hidden_state)
        hidden_state = self.init_hidden_state(self.input_size)
        output, _ = self.output_layer(hidden, hidden_state)
        return output
