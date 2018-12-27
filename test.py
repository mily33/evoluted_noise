import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager
from network import loss_net
import numpy as np

manager = Manager()

def train(args):
    # Construct data_loader, optimizer, etc.
    epoch, rank = args
    model = Net()
    # This will update the shared parameters
    accuracy = 0.1 * epoch
    return accuracy

def test():
    model = loss_net()



if __name__ == '__main__':
    num_processes = 4
    model = loss_net()
    model.share_memory()
    # net = manager.Queue(model)
    multi_pool = Pool(processes=5)
    epoch = 1
    l = manager.Queue(epoch)
    rank = range(5)
    args = ((l, rank[i]) for i in range(5))
    prediction = multi_pool.map_async(train, args)
    print(prediction.get())
