import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=1e-3)
parser.add_argument('--dim', type=int, default=12)
parser.add_argument("--dev", help="device", default="cpu")
parser.add_argument("--epochs", type=int, help="number of epochs for ib", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=512)
parser.add_argument("--ibbeta", type=float, help="beta value of IB", default=0.5)
parser.add_argument("--seed", type=int, help="random seed", default=0)
args = parser.parse_args()

nfile = None
lfile = None
if args.dataset == 'NSL':
    nfile = '../data/nsl.txt'
    lfile = '../data/nsllabel.txt'
elif args.dataset == 'KDD':
    nfile = '../data/kdd.txt'
    lfile = '../data/kddlabel.txt'
elif args.dataset == 'UNSW':
    nfile = '../data/unsw.txt'
    lfile = '../data/unswlabel.txt'
elif args.dataset == 'DOS':
    nfile = '../data/dos.txt'
    lfile = '../data/doslabel.txt'
else:
    df = scipy.io.loadmat('../data/'+args.dataset+".mat")
    numeric = torch.FloatTensor(df['X'])
    labels = (df['y']).astype(float).reshape(-1)


device = torch.device(args.dev)

def compute_distances(x):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    x_t = torch.transpose(x, 0, 1)
    x_t_norm = x_norm.view(1, -1)
    dist = x_norm + x_t_norm - 2.0 * torch.mm(x, x_t)
    dist = torch.clamp(dist, 0, np.inf)

    return dist


def KDE_IXT_estimation(logvar_t, mean_t):
    n_batch, d = mean_t.shape
    var = torch.exp(logvar_t) + 1e-10  # to avoid 0's in the log
    normalization_constant = math.log(n_batch)
    dist = compute_distances(mean_t)
    distance_contribution = -torch.mean(torch.logsumexp(input=-0.5 * dist / var, dim=1))
    I_XT = normalization_constant + distance_contribution

    return I_XT


def get_IXT(mean_t, logvar_t):
    IXT = KDE_IXT_estimation(logvar_t, mean_t)  # in natts
    IXT = IXT / np.log(2)  # in bits
    return IXT


def get_ITY(logits_y, y):
    HY_given_T = ce(logits_y, y)
    ITY = (np.log(2) - HY_given_T) / np.log(2)  # in bits
    return ITY


def get_loss(IXT_upper, ITY_lower):
    loss = -1.0 * (ITY_lower - args.ibbeta * IXT_upper)
    return loss

ce = torch.nn.BCEWithLogitsLoss()
numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
labels = np.loadtxt(lfile, delimiter=',')
if args.dataset == 'KDD':
    labels = 1 - labels
inputdim = numeric.shape[1]

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.e1 = nn.Linear(inputdim, args.dim)
        self.output_layer = nn.Linear(args.dim, 1)

    def forward(self, x):
        mu = self.e1(x)
        intermed = mu + torch.randn_like(mu) * 1
        x = self.output_layer(intermed)
        return x, mu


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
        self.ae = AutoEncoder().to(device)
        self.clock = 0
        self.last_update = -1
        self.updates = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0


    def train_autoencoder(self, data, epochs, labels):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new)
        train_y = Variable(labels).to(device)
        logvar_t = torch.Tensor([0]).to(device)
        for epoch in range(args.epochs):
            self.optimizer.zero_grad()
            train_logits_y, train_mean_t = self.ae(new) #new + 0.001*torch.randn_like(new).to(device)
            train_ITY = get_ITY(train_logits_y, train_y)
            train_IXT = get_IXT(train_mean_t, logvar_t)
            loss = get_loss(train_IXT, train_ITY)
            loss.backward()
            self.optimizer.step()


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
        self.memory = self.ae.e1(new)
        self.memory.requires_grad = False
        self.mem_data = x

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.ae.e1(new)
        loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        self.updates.append(self.update_memory(loss_values, encoder_output, x))
        return loss_values


torch.manual_seed(args.seed)
N = args.memlen
params = {
          'beta': args.beta, 'code_len': args.dim, 'memory_len': N, 'batch_size':1, 'lr':args.lr
         }

model = MemStream(numeric[0].shape[0],params).to(device)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.dim, args.memlen, args.lr, args.epochs)
data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)
model.mem_data = init_data
torch.set_grad_enabled(True)
model.train_autoencoder(numeric[:N].to(device), epochs=args.epochs, labels=torch.Tensor(labels[:N].reshape(-1, 1)))
torch.set_grad_enabled(False)
model.initialize_memory(numeric[labels == 0][:N].to(device))
err = []
for data in data_loader:
    output = model(data.to(device))
    err.append(output)
scores = np.array([i.cpu() for i in err])
auc = metrics.roc_auc_score(labels, scores)
print("ROC-AUC", auc)
