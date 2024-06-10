import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from model import GCN
from torch_geometric.datasets import Planetoid
import random
import numpy as np
import os

random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)


# Training settings
epochs=200
learning_rate=0.01
weight_decay=5e-4
hidden_layer_feature=16
dropout_rate=0.5

dataset = Planetoid(root='/tmp/Cora', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(dataset.num_node_features,hidden_layer_feature,dataset.num_classes).to(device)
data = dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# train
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# test
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')