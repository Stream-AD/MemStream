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
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=1e-5)
parser.add_argument('--dim', type=int, default=12)
parser.add_argument("--dev", help="device", default="cuda:0")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=512)
parser.add_argument("--k", type=int, help="number of nearest neighbours to query", default=3)
parser.add_argument("--gamma", type=float, help="discount factor", default=0)
args = parser.parse_args()  

nfile = None
lfile = None
if args.dataset == 'NSL':
    nfile = '../MemStreamDataset/memnslnumeric.txt'
    lfile = '../MemStreamDataset/memnsllabel.txt'
elif args.dataset == 'KDD': 
    nfile = '../MemStreamDataset/memkddnumeric.txt'
    lfile = '../MemStreamDataset/memkddlabel.txt'
elif args.dataset == 'UNSW': 
    nfile = '../MemStreamDataset/memunswnewnumeric.txt'
    lfile = '../MemStreamDataset/memunswlabel.txt'
elif args.dataset == 'DOS': 
    nfile = '../MemStreamDataset/memdosnewnumeric.txt'
    lfile = '../MemStreamDataset/memdoslabel.txt'
elif args.dataset == 'IDS': 
    nfile = '../icsx_data.csv'
    lfile = '../icsx_label.csv'
elif args.dataset == 'IDS_DOS': 
    nfile = '../icsx_dos_goldeneye_slowloris_numeric.csv'
    lfile = '../icsx_dos_goldeneye_slowloris_label.csv'
elif args.dataset == 'IDS_BOT': 
    nfile = '../icsx_bot_numericnew.csv'
    lfile = '../icsx_bot_label.csv'
elif args.dataset == 'IDS_BRUTE': 
    nfile = '../icsx_brute_numeric.csv'
    lfile = '../icsx_brute_label.csv'
elif args.dataset == 'IDS_DDOS': 
    nfile = '../icsx_ddos_numeric.csv'
    lfile = '../icsx_ddos_label.csv'
elif args.dataset == 'IDS_INF': 
    nfile = '../icsx_infiltration_numeric.csv'
    lfile = '../icsx_infiltration_label.csv'

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
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(device)
        self.clock = 0
        self.last_update = -1
        self.updates = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0
        self.gamma = params['gamma']
        self.K = params['K']
        self.exp = torch.Tensor([self.gamma**i for i in range(self.K)]).to(device)

        
    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.decoder(self.encoder(new + 0.001*torch.randn_like(new).to(device)))
            loss = self.loss_fn(output, new)
            loss.backward()
            self.optimizer.step()
        print("Autoencoder Loss", loss.item()) 


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
        loss_values = (torch.topk(torch.norm(self.memory - encoder_output, dim=1, p=1), k=self.K, largest=False)[0]*self.exp).sum()/self.exp.sum()
        self.updates.append(self.update_memory(loss_values, encoder_output, x))
        return loss_values
    
    
numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
labels = np.loadtxt(lfile, delimiter=',')
if args.dataset == 'KDD':
    labels = 1 - labels
torch.manual_seed(0)
N = args.memlen
params = {
          'beta': args.beta, 'code_len': args.dim, 'memory_len': N, 'batch_size':1, 'lr':args.lr, 'gamma':args.gamma, 'K', args.k
         }

model = MemStream(numeric[0].shape[0],params).to(device)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.dim, args.memlen, args.lr, args.epochs, args.gamma, args.k)
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
np.savetxt('scores.txt', scores)