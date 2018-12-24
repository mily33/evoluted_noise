import argparse
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from torchvision import transforms


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--noise_rate', '-nr', type=float, default=0.2)
parser.add_argument('--noise_type', '-nt', type=str, default='symmetry')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--target_epoch', '-te', type=int, default=10)
args = parser.parse_args()

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
