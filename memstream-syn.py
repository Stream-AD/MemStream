import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
torch.manual_seed(0)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SYN')
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--dim', type=int, default=12)
parser.add_argument("--dev", help="device", default="cpu")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=16)
args = parser.parse_args()  

nfile = None
lfile = None
if args.dataset == 'SYN':
    nfile = './data/syn.txt'
    lfile = './data/synlabel.txt'

device = torch.device(args.dev)
# device = 'cpu'

class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = params['code_len']
        self.memory_len = params['memory_len']
        self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.mem_idx = torch.from_numpy(np.arange(self.memory_len))
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.mem_idx.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.encoder = nn.Identity().to(device)
        self.decoder = nn.Identity().to(device)
        self.clock = 0
        self.last_update = -1
        self.updates = []
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0

        
    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
#         new = (data - self.mean) / self.std
#         new[:, self.std == 0] = 0
#         new = Variable(new)
#         for epoch in range(epochs):
#             self.optimizer.zero_grad()
#             output = self.decoder(self.encoder(new + 0.001*torch.randn_like(new).to(device)))
#             loss = self.loss_fn(output, new)
#             loss.backward()
#             self.optimizer.step()
#         print("Autoencoder Loss", loss.item()) 


    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = torch.argmin(self.mem_idx)
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)    
            self.mem_idx[least_used_pos] = self.count
            self.count += 1
            self.num_mem_update += 1  
            return 1
        return 0    
    
    def initialize_memory(self, x):
        mean, std = model.mem_data.mean(0), model.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        self.memory = self.encoder(new)
        self.memory.requires_grad = False
        self.mem_data = x
    
    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new)
        loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
#         print(loss_values)
#         mem_output = self.memory[idx:idx+1]
#         loss_values = torch.mean(
#             torch.nn.MSELoss(reduction='none')(mem_output, encoder_output), dim=1)
#         for i in range(loss_values.shape[0]):
        self.updates.append(self.update_memory(loss_values, encoder_output, x))
        return loss_values
    
    
numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ',')).reshape(-1, 1)
labels = np.loadtxt(lfile, delimiter=',')
torch.manual_seed(0)
N = args.memlen
params = {
          'beta': args.beta, 'code_len': args.dim, 'memory_len': N, 'batch_size':1, 'lr':args.lr
         }

model = MemStream(numeric[0].shape[0],params).to(device)
model.max_thres=model.max_thres.float()
batch_size = params['batch_size']
print(args.dataset, args.beta, args.dim, args.memlen, args.lr, args.epochs)
data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)
model.mem_data = init_data
torch.set_grad_enabled(True)
model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
torch.set_grad_enabled(False)
model.initialize_memory(Variable(init_data[:N]))
# model.encoder.eval()
err = []
t = time.time()
for data in data_loader:
    output = model(data.to(device))
    err.append(output)
print("Time Taken", time.time() - t)        
scores = np.array([i.cpu() for i in err])
print("Number of Updates: ", model.num_mem_update)
auc = metrics.roc_auc_score(labels, scores)
count = int(np.sum(labels))
preds = np.zeros_like(labels)
indices = np.argsort(scores, axis=0)[::-1]
preds[indices[:count]] = 1
f1 = metrics.f1_score(labels, preds)
print("F1", f1, "AUC", auc)
print("Confusion Matrix", metrics.confusion_matrix(labels, preds))
something = (1 - labels)*scores
something = something[np.nonzero(1-labels)]
normal = np.sort(something)
something = labels*scores
something = something[np.nonzero(labels)]
anomaly = np.sort(something)
print("Normal stats", np.median(normal), np.max(normal), np.min(normal), np.mean(normal))
print("Anomaly stats", np.median(anomaly), np.max(anomaly), np.min(anomaly), np.mean(anomaly))