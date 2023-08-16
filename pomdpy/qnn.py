from torch import Tensor
import torch.nn as nn
import torch.optim as optim
# import torchmetrics


class NN:
    def __init__(self,  num_epochs, learning_rate, layers=2, metric="MSE", L2_reg=0.01):
        self.metric = metric
        self.layer = layers
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.L2_reg = L2_reg
        self.model = nn.Sequential(
            nn.Linear(3072, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        ).to("cuda")

        if (self.metric == "MSE"):
            self.lossfn = nn.MSELoss()
        self.optimizer = optim.SDG(self.model.parameters(
        ), lr=self.learning_rate, weight_decay=self.L2_reg)

        # acc = torchmetrics.Accuracy(n_classes=10).to('cuda')
        costs = []
        accuracies = []
        test_costs = []
        test_accuracies = []
