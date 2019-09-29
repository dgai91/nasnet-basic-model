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


def action_transfer(batch_action):
    kernels = [1, 1, 3, 3, 5, 5, 3, 5]
    filters = [16, 24, 32, 48, 64, 96, 128, 256]
    pool_k = [1, 1, 3, 3, 1, 1, 3, 3]
    batch_action = batch_action.detach().numpy().astype(int)
    batch_action = np.argmax(batch_action, axis=-1)
    batch_action_list = []
    for i in range(batch_action.shape[0]):
        action = [batch_action[0][x:x + 3] for x in range(0, len(batch_action[0]), 3)]
        cnn_ksize = [kernels[c[0]] for c in action]
        num_filters = [filters[c[1]] for c in action]
        max_pool_ksize = [pool_k[c[2]] for c in action]
        batch_action_list.append((cnn_ksize, num_filters, max_pool_ksize))
    return batch_action_list


# train_data = MNIST('../mnist', train=True, transform=ToTensor(), download=False)
# test_data = MNIST('../mnist', train=False, transform=ToTensor())
train_data = CIFAR10('../cifar10', train=True, transform=ToTensor(), download=False)
test_data = CIFAR10('../cifar10', train=False, transform=ToTensor())
print("train_data:", train_data.data.size)
print("test_data:", test_data.data.size)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_layers = 3
num_classes = 8
hidden_dim = 100
param_num = 3 * max_layers
batch_size = 1
reinforce = Reinforce(num_classes, hidden_dim, param_num)
net_manager = NetManager(num_input=32,
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
hidden_state = state.copy()
for i_episode in range(MAX_EPISODES):
    reinforce_optim.zero_grad()
    one_hot_action = reinforce(state, hidden_state, True)
    b_action = action_transfer(one_hot_action)
    rewards, baseline = [], []
    for action in b_action:
        reward = net_manager.get_reward(action)
        rewards.append(reward)
        ema(step, reward)
        baseline.append(EMAs[step])
    rewards = torch.from_numpy(np.array(rewards)).float()
    baseline = torch.from_numpy(np.array(baseline)).float()
    print(rewards.mean())

    scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    prob = reinforce(state, hidden_state, False)
    target = one_hot_action.detach()
    sampler = OneHotCategorical(logits=prob)
    loss = torch.mean(torch.sum(-sampler.log_prob(target), dim=-1) * (rewards - baseline))
    print(loss)
    loss.backward()
    reinforce_optim.step()
