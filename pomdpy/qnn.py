from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch
# import torchmetrics


class NN:
    def __init__(self,  num_epochs, learning_rate, input_size=10, output_size=13,  layers=2, metric="MSE", L2_reg=0, static_interval = 10):
        self.metric = metric
        self.layer = layers
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.L2_reg = L2_reg
        self.static_interval = static_interval
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        ).to("cuda")

        if (self.metric == "MSE"):
            self.loss_function = nn.MSELoss()
        # self.optimizer = optim.SGD(self.model.parameters(
        # ), lr=self.learning_rate, weight_decay=self.L2_reg)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        # acc = torchmetrics.Accuracy(n_classes=10).to('cuda')
        # costs = []
        # accuracies = []
        # test_costs = []
        # test_accuracies = []

    def train(self,  X, y):
        loss_GD = []
#         print(X.dtype , nn.Linear(10, 1, bias=False)
# .weight.dtype)
        X = X.to(torch.float32)
        y = y.to(torch.float32)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        for epoch in range(self.epochs):
            y_pred = self.model(X)
            loss = self.loss_function(y_pred, y)
            loss_GD.append(loss)
            # print(loss_GD[-1])
            loss.backward()
            # if(scheduler.optimizer.param_groups[0]['lr'] >= 0.005): scheduler.step()
            # for static weights to satbilize th enetwork
            if((epoch + 1) % self.static_interval == 0):
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(loss_GD[-1])
    
    def predict(self, X):
        X = X.to(torch.float32)
        return self.model(X)
