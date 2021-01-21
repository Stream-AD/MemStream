# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import time
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
# from torch.autograd import Variable
# torch.manual_seed(0)
np.seterr(divide="ignore", invalid="ignore")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=1e-3)
parser.add_argument('--dim', type=int, default=12)
parser.add_argument("--dev", help="device", default="cpu")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=512)
args = parser.parse_args()  

nfile = None
lfile = None
if args.dataset == 'NSL':
    nfile = './data/nsl.txt'
    lfile = './data/nsllabel.txt'
elif args.dataset == 'KDD': 
    nfile = './data/kdd.txt'
    lfile = './data/kddlabel.txt'
elif args.dataset == 'UNSW': 
    nfile = './data/unsw.txt'
    lfile = './data/unswlabel.txt'
elif args.dataset == 'DOS': 
    nfile = './data/dos.txt'
    lfile = './data/doslabel.txt'
elif args.dataset == 'IDS': 
    nfile = './data/iscx.csv'
    lfile = './data/iscx_label.csv'


# device = torch.device(args.dev)
# device = 'cpu'

class MemStream():
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = params['code_len']
        self.memory_len = params['memory_len']
        self.max_thres = params['beta']
        self.memory = np.random.randn(self.memory_len, self.out_dim)
        self.mem_data = np.random.randn(self.memory_len, self.in_dim)
        self.mem_idx = np.arange(self.memory_len)
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.pca = KernelPCA(n_components=args.dim, kernel='rbf')
        self.clock = 0
        self.last_update = -1
        self.updates = []
        self.count = 0

        
    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        self.pca.fit(new)
        print("PCA trained") 


    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = np.argmin(self.mem_idx)
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
        self.memory = self.pca.transform(new)
        self.mem_data = x
    
    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.pca.transform(new)
        loss_values = np.linalg.norm(self.memory - encoder_output, axis=1, ord=1).min()
        self.updates.append(self.update_memory(loss_values, encoder_output, x))
        return loss_values
    
    
numeric = np.loadtxt(nfile, delimiter = ',')
labels = np.loadtxt(lfile, delimiter=',')
if args.dataset == 'KDD':
    labels = 1 - labels
# torch.manual_seed(0)
np.random.seed(0)
N = args.memlen
params = {
          'beta': args.beta, 'code_len': args.dim, 'memory_len': N, 'batch_size':1
         }

model = MemStream(numeric[0].shape[0],params)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.dim, args.memlen)
# data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N]
model.mem_data = init_data
model.train_autoencoder(init_data, epochs=args.epochs)
model.initialize_memory(init_data[:N])
# model.encoder.eval()
err = []
t = time.time()
for i in range(numeric.shape[0]):
    output = model.forward(numeric[i:i+1])
    err.append(output)
print("Time Taken", time.time() - t)        
scores = np.array(err)
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