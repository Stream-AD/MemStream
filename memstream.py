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
import argparse
import scipy.io
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--beta', type=float, default=1e-5)
parser.add_argument('--dim', type=int, default=12)
parser.add_argument("--dev", help="device", default="cuda:0")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=512)
parser.add_argument("--seed", type=int, help="random seed", default=0)
args = parser.parse_args()  

torch.manual_seed(args.seed)
nfile = None
lfile = None
df = scipy.io.loadmat('/home/siddharth/newdatasets/'+args.dataset)
device = torch.device(args.dev)
# device = 'cpu'
class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = in_dim*2
        self.memory_len = params['memory_len']
        self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0

        
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


    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count%self.memory_len
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)    
            self.count += 1
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
        self.update_memory(loss_values, encoder_output, x)
        return loss_values
    
    
numeric = torch.FloatTensor(df['X'])
labels = (df['y']).astype(float).reshape(-1)
if args.memlen > numeric.shape[0]:
    exit(0)
for beta in [10, 1, 1e-1, 1e-2, 1e-3, 1e-4]:
    args.beta = beta
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
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
    auc = metrics.roc_auc_score(labels, scores)
    count = int(np.sum(labels))
    preds = np.zeros_like(labels)
    indices = np.argsort(scores, axis=0)[::-1]
    preds[indices[:count]] = 1
    f1 = metrics.f1_score(labels, preds)
    print("F1", f1, "AUC", auc)
