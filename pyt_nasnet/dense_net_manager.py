from torch import nn
from torch.optim import Adam
from pyt_nasnet.dense_model import DenseModel
from sklearn.metrics import accuracy_score
import numpy as np


class NetManager:
    def __init__(self, num_input,
                 num_classes,
                 learning_rate,
                 train_loader,
                 test_loader,
                 device,
                 bathc_size=100):

        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.bathc_size = bathc_size
        self.device = device

    def get_reward(self, action):
        model = DenseModel(self.num_input, self.num_classes, action).to(self.device)
        print(model)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for step, (tx, ty) in enumerate(self.train_loader):
            tx = tx.view(-1, 784).to(self.device)
            optimizer.zero_grad()
            to = model(tx)
            loss = loss_func(to, ty.to(self.device))
            loss.backward()
            optimizer.step()
        model.eval()
        mean_val_acc = []
        for step, (tx, ty) in enumerate(self.test_loader):
            tx = tx.view(-1, 784).to(self.device)
            to = model(tx)
            out = np.argmax(to.cpu().detach().numpy(), axis=1)
            val_acc = accuracy_score(ty.cpu().detach().numpy(), out)
            mean_val_acc.append(val_acc)
        return np.mean(mean_val_acc)
