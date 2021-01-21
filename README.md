# MemStream

MemStream augments a representation learning algorithm like PCA, IB and AutoEncoder, with a Memory module to detects anomalies from a multi-aspect data stream in constant time and memory. We output an anomaly score for each record.

## Demo

1. KDD: Run `python3 memstream.py --dataset KDD --beta 0.001 --memlen 512`
2. NSL-KDD: Run `python3 memstream.py --dataset NSL --beta 0.001 --memlen 512`
3. CICIDS-DoS: Run `python3 memstream.py --dataset DOS --beta 0.001 --memlen 1024`
4. UNSW: Run `python3 memstream.py --dataset UNSW --beta 0.1 --memlen 1024`
5. SYN: Run `python3 memstream-syn.py --dataset SYN --beta 1 --memlen 16` 

## Command line options
  * `--dataset`: The dataset to be used for training. Choices 'NSL', 'KDD', 'UNSW', 'DOS' and 'IDS'. (default 'NSL')
  * `--beta`: The threshold beta to be used. (default: 1e-3)
  * `--dim`: The dimensionality of the representations learnt by the representation learning algorithm (default: 12)
  * `--memlen`: The size of the Memory Module (default: 512)
  * `--dev`: Pytorch device to be used for training like "cpu", "cuda:0" etc. (default: 'cpu')
  * `--lr`: Learning rate (default: 0.01)
  * `--epochs`: Number of epochs (default: 5000)

## Input file format
MemStream expects the input multi-aspect record stream to be stored in a contains `,` separated file.

## Datasets
1. [CICIDS-DoS](https://www.unb.ca/cic/datasets/ids-2018.html)
2. [UNSW-NB 15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
3. [KDDCUP99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
4. [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
5. [ISCX2012](https://www.unb.ca/cic/datasets/ids.html) -- Not included in the folder due to space limitation
6. Synthetic Dataset

## Environment
This code has been tested on Debian GNU/Linux 9 with a 12GB Nvidia GeForce RTX 2080 Ti GPU, CUDA Version 10.2 and PyTorch 1.5.