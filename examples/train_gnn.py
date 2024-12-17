import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from load_data import load_data_1, load_data_2, row_normalize
from examples.model import GNN

dataset = 'cora'
feature, label, adj = load_data_2(dataset)
num_classes = len(np.unique(label))
adj = row_normalize(adj)

idx_train, idx_test, _, _ = train_test_split(
    torch.LongTensor(np.arange(label.shape[0])), label, test_size=0.4, random_state=2333)
print(idx_train, idx_test)

adj = torch.FloatTensor(adj)
feature = torch.FloatTensor(feature)
label = torch.LongTensor(label).flatten()

model = GNN(in_size=feature.shape[1], hidden_size=64, out_size=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    output = model(feature, adj)
    loss_train = F.cross_entropy(output[idx_train], label[idx_train])

    _, output = torch.max(output, dim=1)
    acc_train = accuracy_score(label[idx_train].detach().numpy(), output[idx_train])

    print('epoch:{:3d}: | loss:{:1.5f} | acc:{:.3f}'.format(epoch, loss_train, acc_train))
    loss_train.backward()
    optimizer.step()

model.eval()
output = model(feature, adj)

_, output = torch.max(output, dim=1)
acc_test = accuracy_score(label[idx_test].detach().numpy(), output[idx_test])
print(acc_test)

