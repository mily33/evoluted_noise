from tool.soft_target_cross_entropy import SoftCrossEntropy
from data.mnist import MNIST
from torchvision import transforms
import torch
from network import CNN, loss_net
import torch.nn.functional as F
import torch.optim as optim
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from data.utils import noisify
import numpy as np
from tool.utils import Adam


def train(args):
    # Construct data_loader, optimizer, etc.
    epoch, rank = args
    # This will update the shared parameters
    accuracy = 0.1 * rank
    return accuracy


def test_loss_net():
    transform = transforms.ToTensor()
    train_dataset = MNIST(root='/home/datasets/Datasets/mnist/',
                          download=True,
                          train=True,
                          transform=transform,
                          noise_type='symmetry',
                          noise_rate=0.5
                          )
    test_dataset = MNIST(root='/home/datasets/Datasets/mnist/',
                         download=True,
                         train=False,
                         transform=transform,
                         noise_type='symmetry',
                         noise_rate=0.5
                         )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    target_model = CNN(input_channel=1).cuda()
    checkpoint = torch.load('results/mnist_symmetry_0.2/12-27-22:11:05/target_model.pth.tar')
    state_dict = checkpoint['state_dict']
    # print(state_dict.keys())
    for key in list(state_dict.keys()):
        if key[0:7] == 'module.':
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    target_model.load_state_dict(state_dict)
    target_model.eval()

    eloss = loss_net(10).cuda()
    optimizer = optim.Adam(eloss.parameters())
    for epoch in range(100):
        eloss.train()
        for batch_idx, (img, (noisy_target, target)) in enumerate(train_loader):
            img, target = img.cuda(), target.cuda()
            target_output = target_model(img)
            noisy_target = noisy_target.view(-1, 1)
            noisy_target = torch.zeros(img.shape[0], 10).scatter_(1, noisy_target.long(), 1.0)

            varInput = torch.cat((noisy_target.cuda(), target_output), 1)

            output = eloss(varInput)
            optimizer.zero_grad()

            loss = F.nll_loss(torch.log(output), target.squeeze())
            loss.backward()
            optimizer.step()

        eloss.eval()
        correct = 0
        for idx, (data, (noisy_target, target)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            target_output = target_model(data)
            noisy_target = noisy_target.view(-1, 1)
            noisy_target = torch.zeros(data.shape[0], 10).scatter_(1, noisy_target.long(), 1)

            varInput = torch.cat((noisy_target.cuda(), target_output), 1)

            output = eloss(varInput)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        correct = 100. * correct / len(train_loader.dataset)
        print('Epoch %d, correct %f' % (epoch, correct))


def inner_train(x_train, y_noise_train, x_test, y_test, clf, phi):
    x_train = torch.FloatTensor(x_train)
    y_auxiliary = clf(x_train)
    y_noise_train = torch.Tensor(y_noise_train)
    y_noise = torch.zeros(x_train.shape[0], 3).scatter_(1, y_noise_train.long(), 1.0)
    y_input = torch.cat((y_noise, y_auxiliary), 1)

    target = loss_net(3)
    for key in target.state_dict().keys():
        target.state_dict()[key] = target.state_dict()[key] + phi[key]
    target.eval()

    y_input = torch.FloatTensor(y_input)
    y_target = target(y_input)

    f = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 3), torch.nn.Softmax())
    optimizer = optim.SGD(f.parameters(), lr=0.1)
    cross_entropy = SoftCrossEntropy()

    f.train()
    for i in range(60):
        y = f(x_train)
        optimizer.zero_grad()
        ce = cross_entropy.forward(torch.log(y), y_target)
        # print(ce.item())

        ce.backward(retain_graph=True)
        optimizer.step()
    f.eval()
    y = f(torch.FloatTensor(x_test))
    y = np.argmax(y.detach().numpy(), 1)
    # print(sum(y == y_test) / len(y_test))

    return sum(y == y_test) / len(y_test)


def test_iris():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
    y_train_ = np.asarray([[y_train[i]] for i in range(len(y_train))])
    y_train_noisy, actual_noise_rate = noisify(train_labels=y_train_, noise_rate=0.5, noise_type='symmetry', nb_classes=3)
    # clf = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train_noisy)
    # print(clf.score(X_test, y_test))
    f = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 3), torch.nn.Softmax())
    optimizer = optim.SGD(f.parameters(), lr=0.1)
    f.train()
    for i in range(60):
        y = f(torch.FloatTensor(X_train))
        loss = F.nll_loss(torch.log(y), torch.LongTensor(y_train))
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    f.eval()
    y = f(torch.FloatTensor(X_test))
    y = np.argmax(y.detach().numpy(), 1)
    # print(sum(y == y_test)/len(y_test))

    eloss = loss_net(3)
    eloss.eval()
    phi_state_dict = eloss.state_dict()
    for epoch in range(100):
        phi_noise = []
        result = []
        for i in range(10):
            noise = dict()
            for key in eloss.state_dict().keys():
                noise[key] = torch.randn_like(eloss.state_dict()[key]) + phi_state_dict[key]
            phi_noise.append(noise)
            result.append(inner_train(X_train, y_train_noisy, X_test, y_test, f, noise))
        print(np.mean(result))
        for key in eloss.state_dict():
            grad = 0
            for i in range(10):
                grad += result[i] * phi_noise[i][key].cpu()  # - self.outer_l2 * evoluted_loss.state_dict()[key].cpu()
            adam = Adam(shape=grad.shape, stepsize=0.01)
            # print(grad / 10)
            eloss.state_dict()[key] -= 0.1 * grad / 10

    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    # print(scores)


# def target_file()
if __name__ == '__main__':
    test_iris()