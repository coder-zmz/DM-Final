import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from load_data import load_data_1, load_data_2
from examples.model import MLP

dataset = 'cora'
feature, label, _ = load_data_2(dataset)
num_classes = len(np.unique(label))

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, random_state=2333)
print(y_train.shape, y_test.shape)
y_train = y_train.flatten()
y_test = y_test.flatten()

x_train, x_test = [torch.FloatTensor(k) for k in [x_train, x_test]]
y_train, y_test = [torch.LongTensor(k) for k in [y_train, y_test]]
print(y_train, y_test)

model = MLP(in_size=x_train.shape[1], hidden_size=64, out_size=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    output = model(x_train)
    loss_train = F.cross_entropy(output, y_train)

    _, output = torch.max(output, dim=1)
    acc_train = accuracy_score(y_train.detach().numpy(), output)

    print('epoch:{:3d}: | loss:{:1.5f} | acc:{:.3f}'.format(epoch, loss_train, acc_train))
    loss_train.backward()
    optimizer.step()

model.eval()
output = model(x_test)

_, output = torch.max(output, dim=1)
acc_test = accuracy_score(y_test.detach().numpy(), output)
print(acc_test)

