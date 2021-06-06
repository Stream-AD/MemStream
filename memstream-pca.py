import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.spatial as sp
np.seterr(divide="ignore", invalid="ignore")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=1e-3)
parser.add_argument('--dim', type=int, default=12)
parser.add_argument("--memlen", type=int, help="size of memory", default=512)
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
        self.pca = PCA(n_components=args.dim)
        self.clock = 0
        self.last_update = -1
        self.updates = []
        self.count = 0


    def train_autoencoder(self, data):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        self.pca.fit(new)


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
np.random.seed(0)
N = args.memlen
params = {
          'beta': args.beta, 'code_len': args.dim, 'memory_len': N, 'batch_size':1
         }

model = MemStream(numeric[0].shape[0],params)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.dim, args.memlen)
init_data = numeric[labels == 0][:N]
model.mem_data = init_data
model.train_autoencoder(init_data)
model.initialize_memory(init_data[:N])
err = []
for i in range(numeric.shape[0]):
    output = model.forward(numeric[i:i+1])
    err.append(output)
scores = np.array(err)
auc = metrics.roc_auc_score(labels, scores)
print("ROC-AUC", auc)
