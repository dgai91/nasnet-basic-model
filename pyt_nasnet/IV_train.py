from trunk.pyt_nasnet.reg_net_manager import NetManager
from trunk.pyt_nasnet.nas_rnn_anchor import Reinforce, NASCell
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from scipy.stats import norm
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
    loss = torch.FloatTensor([0.0])
    for hp_pred, hp_act in list(zip(all_hp_pred, all_hp_act)):
        target = hp_act.detach()
        sampler = OneHotCategorical(logits=hp_pred)
        l = torch.mean(torch.sum(-sampler.log_prob(target), dim=-1) * (b - r))
        loss += l
    for anchors_pred, anchors_act in list(zip(all_prob_anchors, all_act_anchors)):
        target = anchors_act.detach()
        sampler = Bernoulli(logits=anchors_pred)
        l = torch.mean(torch.sum(-sampler.log_prob(target), dim=-1) * (b - r))
        loss += l
    return loss


def action_transfer(hu_act, anchors):
    units = [16, 32, 64, 128]
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


n, p, ticks = 10000, 20, 1001
X = 2 * np.random.rand(n, p) - 1
Z = np.random.randn(n)
A = np.random.randn(n)  # latent var

W = A + Z  # combined var
Y = 2 * (X[:, 0] <= 0) * A + (X[:, 0] > 0) * W + (1 + (np.sqrt(3) - 1) * (X[:, 0] > 0)) * np.random.randn(n)
X = np.concatenate([X, np.expand_dims(Z, 1), np.expand_dims(W, 1)], axis=1)
X_test = np.zeros((ticks, p + 2))
xvals = np.linspace(-1, 1, ticks)
X_test[:, 0] = xvals
truth = xvals > 0

train_dataset = TensorDataset(torch.from_numpy(X)[400:].float(), torch.from_numpy(Y)[400:].float())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
test_dataset = TensorDataset(torch.from_numpy(X)[:400].float(), torch.from_numpy(Y)[:400].float())
test_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)

device = 'cpu'
param_cate = 4
max_layers = 8
num_param = 2
batch_size = 2
reinforce = Reinforce(NASCell, 20, max_layers, num_param, 35, [param_cate] * 2)
net_manager = NetManager(num_input=p + 2,
                         num_outputs=1,
                         learning_rate=0.001,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         device=device,
                         reg_param=truth,
                         xvals=xvals)

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
exp = 'exp0_episode_'
ALL_STEPS = 2500
state = torch.randn((batch_size, param_cate))
for i_episode in range(ALL_STEPS):
    reinforce_optim.zero_grad()
    hu_act, all_anchors = reinforce.get_action(state)
    b_action, b_anchors = action_transfer(hu_act, all_anchors)
    rewards, baseline = [], []
    for idx, action in enumerate(b_action):
        model_name = exp + str(i_episode) + '_' + str(idx)
        reward = net_manager.get_reward((action, b_anchors[idx]), model_name)
        rewards.append(reward)
        ema(step, reward)
        baseline.append(EMAs[step])
        print('val_loss:', reward)
    rewards = torch.from_numpy(np.array(rewards)).float()
    baseline = torch.from_numpy(np.array(baseline)).float()
    reinforce.store_roll_out(state, rewards)
    scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    pred_logit = reinforce(state)
    state = pred_logit[0][0]
    loss = log_ce_with_pg(pred_logit, (hu_act, all_anchors), rewards, baseline)
    print(loss)
    loss.backward(retain_graph=True)
    reinforce_optim.step()
