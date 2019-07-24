from pyt_nasnet.pyt_net_manager import NetManager
from pyt_nasnet.pyt_nas_rnn import Reinforce, NASCell
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical


# def log_ce_with_pg(pred, truth, r, b):  # (bs, t, c)
#     log_pred = torch.log(pred)
#     batch_loss = torch.sum(torch.mul(truth, log_pred), dim=-1)
#     policy_loss = torch.mul(torch.mean(batch_loss, dim=-1), (b - r))
#     return torch.mean(policy_loss)


def action_transfer(batch_action):
    kernels = [1, 3, 5, 7]
    filters = [24, 36, 48, 64]
    pool_k = [1, 3, 5, 7]
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


train_data = MNIST('../mnist', train=True, transform=ToTensor(), download=False)
test_data = MNIST('../mnist', train=False, transform=ToTensor())
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
device = 'cuda:0'
max_layers = 2
class_num = 4
input_len = 3 * max_layers
batch_size = 30
reinforce = Reinforce(NASCell, 4, input_len, 100)
net_manager = NetManager(num_input=784,
                         num_classes=10,
                         learning_rate=0.001,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         device=device)

MAX_EPISODES = 2500
step = 0

a = 0.5

Prices = []  # prices of everyday
EMAs = []  # ems of everyday


def ema(N, Price):
    Prices.append(Price)
    if N == 0:
        EMAs.append(Price)
    else:
        EMAs.append((1 - a) * EMAs[N - 1] + a * Price)


reinforce_optim = Adam(reinforce.parameters(),
                       lr=0.0006,
                       weight_decay=0.0001)
state = torch.randn((batch_size, class_num))
for i_episode in range(MAX_EPISODES):
    reinforce_optim.zero_grad()
    one_hot_action = reinforce.get_action(state)
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
    reinforce.store_roll_out(state, rewards)

    scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    prob = reinforce(state)[0]
    state = prob[:, -1, :]
    target = one_hot_action.detach()
    sampler = OneHotCategorical(logits=prob)
    loss = torch.mean(torch.sum(-sampler.log_prob(target), dim=-1) * (rewards - baseline))
    print(loss)
    loss.backward(retain_graph=True)
    reinforce_optim.step()
