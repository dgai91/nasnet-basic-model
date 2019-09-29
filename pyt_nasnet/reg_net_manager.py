from torch import nn
from torch.optim import RMSprop
from trunk.pyt_nasnet.dense_model import DenseModel
import numpy as np
import torch
import matplotlib.pyplot as plt


class TiltedLoss(nn.Module):
    def __init__(self, q):
        super(TiltedLoss, self).__init__()
        self.q = torch.FloatTensor(q)

    def forward(self, f, y):
        e = (y.unsqueeze(1).repeat(1, self.q.size()[0]) - f)
        tl = torch.sum(torch.max(torch.mul(e, self.q), torch.mul(e, (self.q - 1))), dim=-1)
        return torch.mean(tl)


class NetManager:
    def __init__(self, num_input,
                 num_outputs,
                 learning_rate,
                 train_loader,
                 test_loader,
                 device,
                 reg_param,
                 xvals):

        self.num_input = num_input
        self.num_classes = num_outputs
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.reg_param = reg_param
        self.xvals = xvals
        X_test = np.zeros((self.xvals.shape[0], self.num_input))
        X_test[:, 0] = self.xvals
        self.X_test = torch.from_numpy(X_test).to(device).float()

    def get_reward(self, action, model_name):
        model = DenseModel(self.num_input, self.num_classes, action).to(self.device)
        if len(self.reg_param) == 3:
            loss_func = TiltedLoss(self.reg_param[0])
        else:
            loss_func = nn.MSELoss()
        optimizer = RMSprop(model.parameters(), lr=self.learning_rate)
        all_mean_val_loss = []
        all_mean_tr_loss = []
        for epoch in range(200):
            model.train()
            model.is_training = True
            mean_tr_loss = []
            for step, (tx, ty) in enumerate(self.train_loader):
                tx = tx.view(-1, self.num_input).to(self.device)
                optimizer.zero_grad()
                to = model(tx)
                loss = loss_func(to, ty.to(self.device))
                loss.backward()
                optimizer.step()
                mean_tr_loss.append(loss.cpu().detach().numpy())
            all_mean_tr_loss.append(np.mean(mean_tr_loss))
        print('train_loss:', -np.mean(all_mean_tr_loss))
        dummy_input = torch.randn(1, 22)
        torch.onnx.export(model, dummy_input, '../saved_model/' + model_name + '.onnx')
        model.eval()
        model.is_training = False
        mean_val_loss = []
        for step, (tx, ty) in enumerate(self.test_loader):
            tx = tx.view(-1, self.num_input).to(self.device)
            to = model(tx)
            l = loss_func(to, ty.to(self.device)).cpu().detach().numpy()
            mean_val_loss.append(l)
        all_mean_val_loss.append(mean_val_loss)
        to = model(self.X_test)
        if len(self.reg_param) == 3:
            q_score, truth_label, colors = self.reg_param
            for idx, (q, t, c) in enumerate(list(zip(q_score, truth_label, colors))):
                plt.plot(self.xvals, to.detach().numpy()[:, idx], label=q, color=c)
                plt.plot(self.xvals, t, label=q, color=c)
        else:
            plt.plot(self.xvals, to.detach().numpy(),)
            plt.plot(self.xvals, self.reg_param)
        plt.legend()
        plt.savefig('../saved_model_graph/' + model_name + '.png')
        plt.show()
        return -np.mean(all_mean_val_loss)
