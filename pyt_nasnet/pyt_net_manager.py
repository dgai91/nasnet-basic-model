from torch import nn
from torch.optim import Adam
from pyt_nasnet.pyt_cnn import CNN
from sklearn.metrics import accuracy_score
import numpy as np


class NetManager:
    def __init__(self, num_input, num_classes, learning_rate, train_loader, test_loader, device,
                 max_step_per_action=5500 * 3,
                 bathc_size=100,
                 dropout_rate=0.85):

        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate
        self.device = device

    def get_reward(self, action):
        model = CNN(self.num_input, self.num_classes, action).to(self.device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for step, (tx, ty) in enumerate(self.train_loader):
            tx = tx.view(-1, 784).unsqueeze(1).to(self.device)
            optimizer.zero_grad()
            to = model(tx)
            loss = loss_func(to, ty.to(self.device))
            loss.backward()
            optimizer.step()
            # if step % 100 == 0:
            #     out = np.argmax(to.cpu().detach().numpy(), axis=1)
            #     y = ty.cpu().detach().numpy()
            #     print("Step " + str(step) +
            #           ", Minibatch Loss= " + "{:.4f}".format(loss) +
            #           ", Current accuracy= " + "{:.3f}".format(accuracy_score(y, out)))
        model.eval()
        mean_val_acc = []
        for step, (tx, ty) in enumerate(self.test_loader):
            tx = tx.view(-1, 784).unsqueeze(1).to(self.device)
            to = model(tx)
            out = np.argmax(to.cpu().detach().numpy(), axis=1)
            val_acc = accuracy_score(ty.cpu().detach().numpy(), out)
            mean_val_acc.append(val_acc)
        return np.mean(mean_val_acc)
