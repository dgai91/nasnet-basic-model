from torch import nn
from torch.optim import Adam
from pyt_nasnet.pyt_cnn import CNN
from sklearn.metrics import accuracy_score
import numpy as np

class NetManager:
    def __init__(self, num_input, num_classes, learning_rate, train_loader, test_loader,
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

    def get_reward(self, action, pre_acc):
        action = action.detach().numpy().astype(int)
        action = [action[0][0][x:x + 4] for x in range(0, len(action[0][0]), 4)]
        model = CNN(self.num_input, self.num_classes, action)
        print(model)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for step, (tx, ty) in enumerate(self.train_loader):
            tx = tx.view(-1, 784).unsqueeze(1)
            optimizer.zero_grad()
            to = model(tx)
            loss = loss_func(to, ty)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                out = np.argmax(to.detach().numpy(), axis=1)
                y = ty.detach().numpy()
                print("Step " + str(step) +
                      ", Minibatch Loss= " + "{:.4f}".format(loss) +
                      ", Current accuracy= " + "{:.3f}".format(accuracy_score(y, out)))
        model.eval()
        for step, (tx, ty) in enumerate(self.train_loader):
            tx = tx.view(-1, 784).unsqueeze(1)
            to = model(tx)
            out = np.argmax(to.detach().numpy(), axis=1)
            acc = accuracy_score(ty.detach().numpy(), out)
            print("!!!!!!acc:", acc, pre_acc)
            if acc - pre_acc <= 0.01:
                return acc, acc
            else:
                return 0.01, acc
