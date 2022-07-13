import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import random
import matplotlib.pyplot as plt

# Negative log (plus small value to avoid 0)
def log_loss(pred):
    return -torch.log(pred + 1e-5)

# ResLayer - code taken from https://github.com/macaodha/geo_prior/blob/dbb366849ecfb6d1586da0efc85561c6d95ff658/geo_prior/models.py
class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out

# Create a network of our desired configuration
def createNet(NUM_CLASS, DIM_INPUT, DIM_HIDDEN):
    class Net(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = DIM_HIDDEN
            self.feats = nn.Sequential(nn.Linear(DIM_INPUT, dim),
                                       nn.ReLU(inplace=True),
                                       ResLayer(dim),
                                       ResLayer(dim),
                                       ResLayer(dim),
                                       ResLayer(dim))
            self.out_proj = nn.Linear(dim, NUM_CLASS)
            self.output_feats = False

        def forward(self, x):
            # If we only want features for transfer learning, we can set output_feats to True
            if self.output_feats:
                return self.feats(x)
            else:
                return torch.softmax(self.out_proj(self.feats(x)), dim=1)

        # Custom loss function as described in report
        def compute_loc_loss(self, y_pred, y_true, y_r):
            loss_pos = log_loss(1.0 - y_pred)  # neg
            for i in range(0, len(y_pred)):
                 loss_pos[i][int(y_true[i])] = NUM_CLASS * log_loss(y_pred[i][int(y_true[i])]) # pos
            loss_bg = log_loss(1.0 - y_r)

            return loss_pos.mean() + loss_bg.mean()

        # Randomises weights
        def random_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal(m.weight)

    return Net(DIM_HIDDEN)

# Code for training the network
def trainNet(net, train_dataloader, x_absences, NUM_CLASS, NUM_ABS, device):

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.tick_params(axis='x', which='both', bottom='false', top='false')
    x = []
    y = []
    count = 0

    for epoch in range(16):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs and labels from the data
            inputs, labels = data
            abs_vals = torch.Tensor(np.empty((len(inputs), inputs.shape[1])).astype(np.float32)).to(device)

            # Get absence values for the data points
            for i in range(0, len(labels)):
                abs_vals[i] = x_absences[random.randint(labels[i].cpu() * NUM_ABS, (labels[i].cpu() * NUM_ABS) + NUM_ABS)]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(inputs.float())

            rand_abs = net(abs_vals)
            loss = net.compute_loc_loss(output, labels, rand_abs)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('Loss after mini-batch %5d: %.5f' %
                      (count, loss.item()))
                # x.append((i+1)*count)
                # y.append(loss.item())
                running_loss = 0.0
                count+=1
        x.append(epoch)
        y.append(epoch_loss)
    print('Finished Training')
    # Plot loss over time
    plt.plot(x, y)
    plt.show()