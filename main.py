from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from trainer import OuterTrainer
from network import CNN


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('Epoch {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     epoch, test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    from config import args
    from config import train_dataset
    from config import test_dataset
    from config import fun_params

    args = args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = train_dataset
    test_dataset = test_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    target_model = CNN(input_channel=1)
    target_model = torch.nn.DataParallel(target_model).cuda()
    target_optimizer = optim.SGD(target_model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.target_epoch):
        train(args, target_model, device, train_loader, target_optimizer, epoch)

    test_loss, correct = test(target_model, test_loader)
    print('Target model completed. Test loss %f, test accuracy %f' % (test_loss, correct))
    trainer = OuterTrainer(target_model, device, train_dataset, test_dataset, fun_params)
    trainer.train(fun_params['outer_n_worker'])


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')
    main()
