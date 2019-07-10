from pyt_nasnet.pyt_net_manager import NetManager
from pyt_nasnet.pyt_reinforce import Reinforce
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from pyt_nasnet.pyt_optimizer import DCRMSprop
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
import torch.nn.functional as F

train_data = MNIST('../mnist', train=True, transform=ToTensor(), download=False)
test_data = MNIST('../mnist', train=False, transform=ToTensor())
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
max_layers = 3


def softmax_cross_entropy_with_logits(output, target):
    prob_out = F.softmax(output, -1)
    prob_target = F.softmax(target, -1)
    return torch.mean(torch.mul(prob_target, torch.log(prob_out)))


reinforce = Reinforce(max_layers, 1)
net_manager = NetManager(num_input=784,
                         num_classes=10,
                         learning_rate=0.001,
                         train_loader=train_loader,
                         test_loader=test_loader)

MAX_EPISODES = 2500
step = 0
state = np.array([[10.0, 128.0, 1.0, 1.0] * max_layers], dtype=np.float32)
state = torch.from_numpy(state)
pre_acc = 0.0
total_rewards = 0
for i_episode in range(MAX_EPISODES):
    action = reinforce.get_action(state)
    if all(ai > 0 for ai in action[0][0]):
        reward, pre_acc = net_manager.get_reward(action, pre_acc)
        print("=====>", reward, pre_acc)
    else:
        reward = -1.0
    total_rewards += reward
    state = action[0] / reinforce.division_rate
    reinforce.store_roll_out(state, reward)
    reinforce_optim = DCRMSprop(reinforce.parameters(),
                                lr=0.99,
                                discounted_rewards=reward,
                                weight_decay=0.001)
    scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    log_prob = reinforce(state)
    target = state.detach()
    loss = softmax_cross_entropy_with_logits(log_prob, target)
    reinforce_optim.zero_grad()
    loss.backward(retain_graph=True)
    reinforce_optim.step()
