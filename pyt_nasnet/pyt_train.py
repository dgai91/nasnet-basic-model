from trunk.pyt_nasnet.pyt_net_manager import NetManager
from trunk.pyt_nasnet.pyt_nas_rnn import Reinforce
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical


def action_instantiation(batch_action, param_list):
    batch = batch_action[0].detach().size()[0]
    batch_action_list = []
    for idx in range(batch):
        action, trans_acts = [], []
        for x in range(0, len(batch_action), len(param_list)):
            action.append([batch_action[x + i][idx] for i in range(len(param_list))])
        for param_id, param in enumerate(param_list):
            trans_acts.append([param[act[param_id]] for act in action])
        batch_action_list.append(tuple(trans_acts))
    return batch_action_list


train_data = MNIST('../mnist', train=True, transform=ToTensor(), download=False)
test_data = MNIST('../mnist', train=False, transform=ToTensor())
# train_data = CIFAR10('../cifar10', train=True, transform=ToTensor(), download=False)
# test_data = CIFAR10('../cifar10', train=False, transform=ToTensor())
print("train_data:", train_data.data.size)
print("test_data:", test_data.data.size)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
KERNELS = [1, 3, 5, 7]
FILTERS = [16, 32, 64, 128, 256]
POOLING_SIZE = [1, 3, 5]
max_layers = 3
num_classes = 8
hidden_dim = 100
all_params = [len(KERNELS), len(FILTERS), len(POOLING_SIZE)]
all_param_list = [KERNELS, FILTERS, POOLING_SIZE]
batch_size = 1
reinforce = Reinforce(hidden_dim, all_params, max_layers)
net_manager = NetManager(num_input=28,
                         in_channel=1,
                         num_classes=10,
                         learning_rate=0.001,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         device=device)

MAX_EPISODES, step, a = 2500, 0, 0.5
Prices, EMAs = [], []  # prices, ems of everyday,


def ema(N, Price):
    Prices.append(Price)
    if N == 0:
        EMAs.append(Price)
    else:
        EMAs.append((1 - a) * EMAs[N - 1] + a * Price)


reinforce_optim = Adam(reinforce.parameters(),
                       lr=0.0006,
                       weight_decay=0.0001)
state = torch.zeros((batch_size, hidden_dim))
hidden_state = (state.clone(), state.clone())
for i_episode in range(MAX_EPISODES):

    one_hot_action, log_probs = reinforce(state, hidden_state)
    b_action = action_instantiation(one_hot_action, all_param_list)
    rewards, baseline = [], []
    for action in b_action:
        reward = net_manager.get_reward(action)
        rewards.append(reward)
        ema(step, reward)
        baseline.append(EMAs[step])
    rewards = torch.from_numpy(np.array(rewards)).float()
    baseline = torch.from_numpy(np.array(baseline)).float()
    print(rewards.mean())
    # scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    print(log_probs)
    loss = torch.mean(torch.sum(-log_probs, dim=-1) * (rewards - baseline))
    reinforce_optim.zero_grad()
    print(loss)
    loss.backward()
    reinforce_optim.step()
