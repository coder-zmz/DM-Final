from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=3, dropout=0.5):
        super(MLP, self).__init__()

        self.layer_1 = nn.Linear(in_size, hidden_size, bias=True)
        self.layer_2 = nn.Linear(hidden_size, out_size, bias=True)

        '''
        if num_layers == 1:
            hidden_size = out_size

        self.pipeline = nn.Sequential(OrderedDict([
            ('layer_0', nn.Linear(in_size, hidden_size, bias=(num_layers != 1))),
            ('dropout_0', nn.Dropout(dropout)),
            ('relu_0', nn.ReLU())
        ]))

        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, out_size, bias=True))
            else:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, hidden_size, bias=True))
                self.pipeline.add_module('dropout_{}'.format(i), nn.Dropout(dropout))
                self.pipeline.add_module('relu_{}'.format(i), nn.ReLU())
        '''

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature):
        '''
        return F.softmax(self.pipeline(feature), dim=1)
        '''
        h = F.relu(self.layer_1(feature))
        out = self.layer_2(h)
        return out


class GNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(GNN, self).__init__()
        self.layer_1 = nn.Linear(in_size, hidden_size, bias=True)
        self.layer_2 = nn.Linear(hidden_size, out_size, bias=True)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, adj):
        x_agg = torch.mm(adj, x)
        h = F.relu(self.layer_1(x_agg))

        h_agg = torch.mm(adj, h)
        out = self.layer_2(h_agg)

        return out

