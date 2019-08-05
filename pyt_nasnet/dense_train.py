from pyt_nasnet.dense_net_manager import NetManager
from pyt_nasnet.nas_rnn_anchor import Reinforce, NASCell
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.bernoulli import Bernoulli
seed = np.random.seed(1111)


def log_ce_with_pg(pred, truth, r, b):  # (bs, t, c)
    all_hp_pred, all_prob_anchors = pred
    all_hp_act, all_act_anchors = truth
    loss = 0
    for hp_pred, hp_act in list(zip(all_hp_pred, all_hp_act)):
        target = hp_act.detach()
        sampler = OneHotCategorical(logits=hp_pred)
        l = torch.mean(torch.sum(-sampler.log_prob(target), dim=-1) * (r - b))
        loss += l
    for anchors_pred, anchors_act in list(zip(all_prob_anchors, all_act_anchors)):
        target = anchors_act.detach()
        sampler = Bernoulli(logits=anchors_pred)
        l = torch.mean(torch.sum(-sampler.log_prob(target), dim=-1) * (r - b))
        loss += l
    return loss


def action_transfer(hu_act, anchors):
    units = [16, 32, 64, 128, 256]
    hu_act = [x.detach().numpy().astype(int) for x in hu_act]
    hu_act = np.argmax(np.stack(hu_act, axis=1), axis=-1)
    batch_action_list, batch_anchor_list = [], []
    for i in range(hu_act.shape[0]):
        act_list = hu_act[i].tolist()
        action = [units[t] for t in act_list]
        batch_action_list.append(action)
        anchor_list = []
        for layer_id in range(len(anchors)):
            anchor = anchors[layer_id].detach().numpy().astype(int)[i]
            anchor_list.append(anchor.tolist())
        batch_anchor_list.append(anchor_list)
    return batch_action_list, batch_anchor_list


train_data = CIFAR10('../cifar10', train=True, transform=ToTensor(), download=True)
test_data = CIFAR10('../cifar10', train=False, transform=ToTensor())
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
device = 'cpu'
max_layers = 3
class_num = 4
num_param = 2
batch_size = 15
reinforce = Reinforce(NASCell, 20, max_layers, num_param, 35, [4, 4])
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
    hu_act, all_anchors = reinforce.get_action(state)
    b_action, b_anchors = action_transfer(hu_act, all_anchors)
    rewards, baseline = [], []
    for idx, action in enumerate(b_action):
        reward = net_manager.get_reward((action, b_anchors[idx]))
        print(reward)
        rewards.append(reward)
        ema(step, reward)
        baseline.append(EMAs[step])
    rewards = torch.from_numpy(np.array(rewards)).float()
    baseline = torch.from_numpy(np.array(baseline)).float()
    print(rewards.mean())
    reinforce.store_roll_out(state, rewards)

    scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    pred_logit = reinforce(state)
    state = pred_logit[0][0]
    loss = log_ce_with_pg(pred_logit, (hu_act, all_anchors), rewards, baseline)
    print(loss)
    loss.backward(retain_graph=True)
    reinforce_optim.step()
