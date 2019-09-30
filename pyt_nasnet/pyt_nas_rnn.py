import torch
import torch.nn as nn
import torch.nn.functional as F


class Reinforce(nn.Module):
    def __init__(self, hidden_size, all_params, num_layers):
        super(Reinforce, self).__init__()
        self.all_params = all_params
        self.num_layers = num_layers
        self.embedding = nn.Embedding(sum(all_params), hidden_size)
        self.rnn_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.decoders = nn.ModuleList([nn.Linear(hidden_size, params) for params in all_params])

    def call_rnn(self, inputs, param_id, hidden_states=None):
        embed = self.embedding(inputs) if param_id != 0 else inputs
        encoded, hidden_states = self.rnn_cell(embed, hidden_states)
        output = self.decoders[param_id](encoded)
        return output, hidden_states

    def forward(self, inputs, hidden_states, is_sample=False):
        outputs = []
        for layer_id in range(len(self.num_layers)):
            for param_id in range(len(self.all_params)):
                output, hidden_states = self.call_rnn(inputs, param_id, hidden_states)
                action_prob = F.softmax(output, -1)
                inputs = action_prob.multinomial(num_samples=1)
                inputs += sum(self.all_params[:param_id - 1])
                outputs.append(inputs) if is_sample else outputs.append(output)
        return torch.stack(tuple(outputs), dim=1)  # (bs, T, 1) or (bs, T, od)
