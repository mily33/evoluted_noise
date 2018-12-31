import torch
from tool.logz import G
import numpy as np
from network import loss_net
from network import CNN
import time
import os
import torch.optim as optim
from torch.multiprocessing import Pool
import torch.nn.functional as F
from tool.utils import PiecewiseSchedule
from tool.utils import Adam
from tool.soft_target_cross_entropy import SoftCrossEntropy


def inner_train(args):
    epoch, pool_rank, phi, target_model, train_dataset, test_dataset, inner_args = args
    trainer = InnerTrainer(target_model, train_dataset, test_dataset, inner_args, pool_rank)
    accuracy = trainer.train(epoch, pool_rank, phi)
    # accuracy = 0.1 * pool_rank
    print('Worker %d, test accuracy: %f' % (pool_rank, accuracy))
    return dict(accuracy=accuracy)


class InnerTrainer(object):
    def __init__(self, target_model, train_dataset, test_dataset, args, pool_rank=None):
        self.target_model = target_model
        self.test_batch_size = args['test_batch_size']
        self.logdir = args['logdir']
        self.save_model = False
        self.log = G()
        self.nclass = args['nclass']
        if self.save_model:
            self.logdir = os.path.join(self.logdir, '%d' % pool_rank)
            if not os.path.exists(self.logdir):
                os.mkdir(self.logdir)
            self.log.configure_output_dir(self.logdir)
            self.log.save_params(args)
            self.log.path_model = os.path.join(self.logdir, 'model%d.pth.tar')

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['inner_batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=True)

        self.inner_epoch_freq = args['inner_epoch_freq']
        self.inner_batch_size = args['inner_batch_size']
        self.inner_lr = args['inner_lr']
        self.inner_momentum = args['inner_momentum']

    def train(self, epoch, pool_rank, phi):
        model = CNN(input_channel=1)
        model = torch.nn.DataParallel(model).cuda()
        optimizer = optim.SGD(model.parameters(), lr=self.inner_lr, momentum=self.inner_momentum)
        cross_entropy = SoftCrossEntropy()
        model.train()

        self.target_model.eval()

        eloss = loss_net(self.nclass)
        eloss = torch.nn.DataParallel(eloss).cuda()
        for key in eloss.state_dict().keys():
            eloss.state_dict()[key] = eloss.state_dict()[key] + phi[key]
        eloss.eval()
        minloss = 10000
        # print('start train model')
        for e in range(self.inner_epoch_freq):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.cuda()
                optimizer.zero_grad()

                output = model(data)
                target_output = self.target_model(data)
                target = target.view(-1, 1)

                target = torch.zeros(data.shape[0], 10).scatter_(1, target.long(), 1.0)
                varInput = torch.cat((target.cuda(), target_output), 1)
                # print(varInput.shape, target_output.shape)
                soft_target = eloss.forward(varInput)
                # print(soft_target)
                loss = cross_entropy(output, soft_target)
                if self.save_model:
                    if epoch % 20 == 0:
                        self.log.log_tabular('epoch', batch_idx)
                        self.log.log_tabular('loss', loss.item())
                    if loss < minloss:
                        torch.save({'epoch': batch_idx + 1, 'state_dict': model.state_dict(), 'best_loss': minloss,
                                    'optimizer': optimizer.state_dict()}, self.log.path_model)
                loss.backward()
                optimizer.step()
                # print('[pool: %d] epoch %d, loss: %f' % (pool_rank, epoch, loss.item()))

            if epoch % 20 == 0 and self.save_model:
                self.log.dump_tabular()

        accuracy = self.test(model)
        return accuracy

    def test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.nll_loss(output, target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        return 10 * correct / len(self.test_loader.dataset)


class OuterTrainer(object):
    def __init__(self, target_model, device, train_dataset, test_dataset, args):
        self.target_model = target_model
        self.device = device
        self.test_batch_size = args['test_batch_size']
        self.logdir = args['logdir']
        self.save_model = args['save_model']
        self.log = G()
        if self.save_model:
            self.log.configure_output_dir(self.logdir)
            self.log.save_params(args)
            self.log.path_model = os.path.join(self.logdir, 'model.pth.tar')

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.inner_args = args['inner_args']

        self.outer_n_worker = args['outer_n_worker']
        self.outer_lr = args['outer_lr']
        self.outer_n_epoch = args['outer_n_epoch']
        self.outer_std = args['outer_std']
        self.outer_l2 = args['outer_l2']
        self.outer_n_noise = args['outer_n_noise']

    @staticmethod
    def save_theta(theta, dir, ext=''):
        save_path = os.path.join(dir, 'thetas')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'theta{}.npy'.format(ext)), theta)

    @staticmethod
    def load_theta(path):
        return np.load(path)

    def train(self, n_process=None):
        assert n_process is not None
        pool = Pool(processes=n_process)
        outer_lr_scheduler = PiecewiseSchedule([(0, self.outer_lr),
                                                (int(self.outer_n_epoch / 2), self.outer_lr * 0.1)],
                                               outside_value=self.outer_lr * 0.1)
        evoluted_loss = loss_net(self.inner_args['nclass'])
        evoluted_loss = torch.nn.DataParallel(evoluted_loss).cuda()

        evoluted_loss.train()
        for epoch in range(self.outer_n_epoch):
            phi_state_dict = evoluted_loss.state_dict()
            phi_noise = []
            for i in range(self.outer_n_noise):
                noise = dict()
                for key in evoluted_loss.state_dict().keys():
                    noise[key] = self.outer_std * torch.randn_like(evoluted_loss.state_dict()[key]) + phi_state_dict[key]
                phi_noise.append(noise)
            start_time = time.time()

            assert self.outer_n_worker % self.outer_n_noise == 0

            args = ((epoch, i, phi_noise[i], self.target_model, self.train_dataset, self.test_dataset, self.inner_args)
                    for i in range(self.outer_n_noise))
            results = pool.map_async(inner_train, args)
            results = results.get()
            result = [np.mean(r['accuracy']) for r in results]

            print('Epoch %d, mean accuracy: %f, max accuracy: %f(%d), min accuracy: %f(%d)' %
                  (epoch, np.mean(result), np.max(result), result.index(np.max(result)), np.min(result), result.index(np.min(result))))

            print('All inner loops completed, returns gathered ({:.2f} sec).'.format(time.time() - start_time))
            print('\n')
            for key in evoluted_loss.state_dict():
                grad = 0
                for i in range(self.outer_n_noise):
                    grad += result[i] * phi_noise[i][key].cpu() # - self.outer_l2 * evoluted_loss.state_dict()[key].cpu()
                lr = outer_lr_scheduler.value(epoch)
                adam = Adam(shape=grad.shape, stepsize=lr)
                evoluted_loss.state_dict()[key] -= adam.step(grad / self.outer_n_worker).cuda()

            if self.save_model:
                torch.save({'epoch': epoch + 1, 'state_dict': evoluted_loss.state_dict()}, self.log.path_model)
                self.log.log_tabular('epoch', epoch)
                for i in range(len(result)):
                    self.log.log_tabular('reward %d' % i, result[i])
                self.log.dump_tabular()
        pool.close()

    def test(self, model_path):
        model = CNN(input_channel=1)
        model = torch.nn.DataParallel(model).cuda()
        optimizer = optim.SGD(model.parameters(), lr=self.inner_args['inner_lr'], momentum=self.inner_args['inner_momentum'])
        cross_entropy = SoftCrossEntropy()
        model.train()

        self.target_model.eval()

        eloss = loss_net(self.inner_args['nclass'])
        eloss = torch.nn.DataParallel(eloss).cuda()

        assert os.path.exists(model_path), 'model path is not exist.'
        check_point = torch.load(model_path)
        eloss.load_state_dict(check_point['state_dict'])

        eloss.eval()
        minloss = 10000
        # print('start train model')
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.inner_args['inner_batch_size'],
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.inner_args['test_batch_size'], shuffle=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            optimizer.zero_grad()

            output = model(data)
            target_output = self.target_model(data)
            target = target.view(-1, 1)

            target = torch.zeros(data.shape[0], 10).scatter_(1, target.long(), 1.0)
            varInput = torch.cat((target.cuda(), target_output), 1)
            # print(varInput.shape, target_output.shape)
            soft_target = eloss.forward(varInput)
            # print(soft_target)
            loss = cross_entropy(output, soft_target)
            if self.save_model:
                self.log.log_tabular('epoch', batch_idx)
                self.log.log_tabular('loss', loss.item())
                self.log.dump_tabular()
                if loss < minloss:
                    torch.save({'epoch': batch_idx + 1, 'state_dict': model.state_dict(), 'best_loss': minloss,
                                'optimizer': optimizer.state_dict()}, self.log.path_model)
            loss.backward()
            optimizer.step()

