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

    def call_rnn(self, inputs, param_id, layer_id, hidden_states):
        embed = self.embedding(inputs).squeeze(1) if param_id + layer_id != 0 else inputs
        hidden_states = self.rnn_cell(embed, hidden_states)
        output = self.decoders[param_id](hidden_states[0])
        return output, hidden_states

    def forward(self, inputs, hidden_states, is_sample=False):
        outputs = []
        for layer_id in range(self.num_layers):
            for param_id in range(len(self.all_params)):
                output, hidden_states = self.call_rnn(inputs, param_id, layer_id, hidden_states)
                action_prob = F.softmax(output, -1)
                action = action_prob.multinomial(num_samples=1).detach()
                log_prob = F.log_softmax(output, dim=-1)
                selected_log_prob = log_prob.gather(1, action.clone().detach())
                outputs.append(action) if is_sample else outputs.append(selected_log_prob[:, 0])
                inputs = action + sum(self.all_params[:param_id - 1]) if param_id > 0 else action
        return torch.cat(tuple(outputs))  # (bs, T, 1) or (bs, T, od)
