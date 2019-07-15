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


def softmax_cross_entropy_with_logits(output, target):
    prob_out = F.softmax(output, -1)
    prob_target = F.softmax(target, -1)
    return torch.mul(torch.mean(torch.mul(prob_target, torch.log(prob_out))), -1)


def action_transfer(action):
    kernels = [1, 3, 5, 7]
    filters = [24, 36, 48, 64]
    pool_k = [1, 3, 5, 7]
    action = action.detach().numpy().astype(int)
    action = np.argmax(action, axis=-1)
    action = [action[0][x:x + 3] for x in range(0, len(action[0]), 3)]
    cnn_ksize = [kernels[c[0]] for c in action]
    num_filters = [filters[c[1]] for c in action]
    max_pool_ksize = [pool_k[c[2]] for c in action]
    return cnn_ksize, num_filters, max_pool_ksize


train_data = MNIST('../mnist', train=True, transform=ToTensor(), download=False)
test_data = MNIST('../mnist', train=False, transform=ToTensor())
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
max_layers = 2

reinforce = Reinforce(max_layers, 4)
net_manager = NetManager(num_input=784,
                         num_classes=10,
                         learning_rate=0.001,
                         train_loader=train_loader,
                         test_loader=test_loader)

class_num = 4
input_len = 3 * max_layers
label = torch.LongTensor(input_len, 1).random_() % class_num
state = torch.zeros(input_len, class_num).scatter_(1, label, 1).unsqueeze(0)
state = state.float()
MAX_EPISODES = 2500
step = 0
for i_episode in range(MAX_EPISODES):
    state = reinforce.get_action(state)
    action = action_transfer(state)

    reward = net_manager.get_reward(action)

    reinforce.store_roll_out(state, reward)
    baseline = np.mean(reinforce.reward_buffer)
    reinforce_optim = DCRMSprop(reinforce.parameters(),
                                lr=0.99,
                                discounted_rewards=reward,
                                baseline=baseline,
                                weight_decay=0.001)
    scheduler = StepLR(reinforce_optim, step_size=500, gamma=0.96)
    log_prob = reinforce(state)
    target = state.detach()
    loss = softmax_cross_entropy_with_logits(log_prob, target)
    print(loss)
    reinforce_optim.zero_grad()
    loss.backward(retain_graph=True)
    reinforce_optim.step()
