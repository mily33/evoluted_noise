import argparse
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from torchvision import transforms
import time
import os


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--noise_rate', '-nr', type=float, default=0.2)
parser.add_argument('--noise_type', '-nt', type=str, default='symmetry')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--target_epoch', '-te', type=int, default=1)
args = parser.parse_args()
nclass = 0
transform = transforms.ToTensor()
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
if args.dataset == 'mnist':
    train_dataset = MNIST(root='/home/datasets/Datasets/mnist/',
                          download=True,
                          train=True,
                          transform=transform,
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )
    test_dataset = MNIST(root='/home/datasets/Datasets/mnist/',
                         download=True,
                         train=False,
                         transform=transform
                         )
    nclass = 10
elif args.dataset == 'cifar10':
    train_dataset = CIFAR10(root='./data/',
                            download=True,
                            train=True,
                            transform=transform,
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    test_dataset = CIFAR10(root='./data/',
                           download=True,
                           train=False,
                           transform=transform,
                           )
    nclass = 10
elif args.dataset == 'cifar100':
    train_dataset = CIFAR100(root='./data/',
                             download=True,
                             train=True,
                             transform=transform,
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate
                             )

    test_dataset = CIFAR100(root='./data/',
                            download=True,
                            train=False,
                            transform=transform
                            )
    nclass = 100

logdir = None
if args.save_model:
    path = os.path.join('results', args.dataset + '_' + args.noise_type + '_' + str(args.noise_rate))
    if not (os.path.exists(path)):
        os.makedirs(path)
    logdir = os.path.join(path, time.strftime('%m-%d-%H:%M:%S'))

fun_params = dict(test_batch_size=args.test_batch_size,
                  logdir=logdir,
                  outer_n_worker=7,
                  outer_lr=1e-2,
                  outer_n_epoch=50,
                  outer_std=0.01,
                  outer_l2=0.001,
                  save_model=args.save_model,
                  inner_args=dict(inner_epoch_freq=20,
                                  test_batch_size=args.test_batch_size,
                                  inner_batch_size=64,
                                  inner_lr=args.lr,
                                  inner_momentum=args.momentum,
                                  save_model=args.save_model,
                                  logdir=logdir,
                                  nclass=nclass))
