from torch.multiprocessing import Pool, Manager


manager = Manager()


def train(args):
    # Construct data_loader, optimizer, etc.
    epoch, rank = args
    # This will update the shared parameters
    accuracy = 0.1 * rank
    return accuracy


if __name__ == '__main__':
    multi_pool = Pool(processes=5)
    epoch = 1
    l = manager.Queue(epoch)
    rank = range(10)
    args = ((l, rank[i]) for i in range(10))
    prediction = multi_pool.map_async(train, args)
    print(prediction.get())
