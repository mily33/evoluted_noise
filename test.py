import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager
from network import Net

manager = Manager()

def train(args):
    # Construct data_loader, optimizer, etc.
    print(type(args), len(args))
    model = Net()
    # This will update the shared parameters
    accuracy = 0.1 
    return accuracy


if __name__ == '__main__':
    num_processes = 4
    model = Net()
    model.share_memory()
    # net = manager.Queue(model)
    multi_pool = Pool(processes=5)
    epoch = [1] * 5
    rank = range(5)
    prediction = multi_pool.map_async(train, (epoch, rank))
    print(prediction.get())
