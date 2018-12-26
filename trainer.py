import torch
from tool.logz import G
import numpy as np
from network import loss_net
from network import CNN
from mpi4py import MPI
import time
import os
import torch.optim as optim
from network import Net
from torch.multiprocessing import Pool
import torch.nn.functional as F
from tool.utils import PiecewiseSchedule, Adam, relative_ranks


def inner_train(args):
    print(len(args))
    epoch, pool_rank, phi, trainer = args
    accuracy = trainer.train(epoch, pool_rank, phi)
    print('Worker %d, test accuracy: %f' % (pool_rank, accuracy))
    return dict(accuracy=accuracy)


class InnerTrainer(object):
    def __init__(self, target_model, train_dataset, test_dataset, args):
        self.target_model = target_model
        self.test_batch_size = args['test_batch_size']
        self.logdir = args['logdir']
        self.save_model = args['save_model']
        self.log = G()
        if self.save_model:
            self.log.configure_output_dir(self.logdir)
            self.log.save_params(args)
            self.log.path_model = self.logdir

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['inner_batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=True)

        self.inner_epoch_freq = args['inner_epoch_freq']
        self.inner_batch_size = args['inner_batch_size']
        self.inner_lr = args['inner_lr']
        self.inner_momentum = args['inner_momentum']

    def train(self, epoch, pool_rank, phi):
        if epoch % 20 == 0 and self.save_model:
            logdir = os.path.join(self.logdir, '%d' % pool_rank)
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log = G()
            log.configure_output_dir(logdir)
            log.path_model = os.path.join(logdir, 'model%d.pth.tar' % epoch)

        model = CNN(input_channel=1)
        model = torch.nn.DataParallel(model).cuda()
        optimizer = optim.SGD(model.parameters(), lr=self.inner_lr, momentum=self.inner_momentum)
        model.train()
        eloss = loss_net()
        eloss.set_params_1d(phi)
        minloss = 10000
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            target_output = self.target_model(data)
            soft_target = eloss.forward(torch.cat((target, target_output)))
            loss = F.nll_loss(output, soft_target)
            if self.save_model:
                if epoch % 20 == 0:
                    log.log_tabular('epoch', batch_idx)
                    log.log_tabular('loss', loss)
                if loss < minloss:
                    torch.save({'epoch': batch_idx + 1, 'state_dict': model.state_dict(), 'best_loss': minloss,
                                'optimizer': optimizer.state_dict()}, log.path_model)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 and self.save_model:
            log.dump_tabular()
            log.pickle_tf_vars()

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
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        return 100. * correct / len(self.test_loader.dataset)


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
            self.log.path_model = self.logdir

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.inner_args = args['inner_args']

        self.outer_n_worker = args['outer_n_worker']
        self.outer_lr = args['outer_lr']
        self.outer_n_epoch = args['outer_n_epoch']
        self.outer_std = args['outer_std']
        self.outer_l2 = args['outer_l2']

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
        evoluted_loss = loss_net()
        phi = evoluted_loss.get_params_1d()
        num_params = len(phi)
        adam = Adam(shape=(num_params, ), beta1=0., stepsize=self.outer_lr, dtype=np.float32)

        for epoch in range(self.outer_n_epoch):
            adam.stepsize = outer_lr_scheduler.value(epoch)

            noise = np.random.randn(self.outer_n_worker, num_params)
            phi_noise = phi[np.newaxis, :] + noise * self.outer_std
            phi_noise = phi_noise.reshape(MPI.COMM_WORLD.Get_size(), -1)

            recvbuf = np.empty(phi_noise.shape[1], dtype='float')
            MPI.COMM_WORLD.Scatter(phi_noise, recvbuf, root=0)
            phi_noise = recvbuf.reshape(-1, num_params)

            start_time = time.time()
            pool_size = int(self.outer_n_worker / MPI.COMM_WORLD.Get_size())

            trainer = InnerTrainer(self.target_model, self.train_dataset, self.test_dataset, self.inner_args)
            results = pool.map_async(inner_train,
                                     ([epoch] * pool_size, range(pool_size), phi_noise, [trainer]*pool_size))
            results = results.get()
            result = [np.mean(r['accuracy']) for r in results]

            recvbuf = np.empty([MPI.COMM_WORLD.Get_size(), 7 * pool_size],
                               # 7 = number of scalars in results vector
                               dtype='float') if MPI.COMM_WORLD.Get_rank() == 0 else None
            results_processed_arr = np.asarray([result], dtype='float').ravel()
            MPI.COMM_WORLD.Gather(results_processed_arr, recvbuf, root=0)

            print('Epoch %d, max accuracy: %f(%d), min accuracy: %f(%d)' %
                  (epoch, np.max(result), result.index(np.max(result)), np.min(result), result.index(np.min(result))))

            if self.save_model:
                self.log.log_tabular('epoch', epoch)
                for i in range(len(result)):
                    self.log.log_tabular('reward %d' % i, result[0])

            # outer loop  update at node 0
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('All inner loops completed, returns gathered ({:.2f} sec).'.format(time.time() - start_time))
                results_processed_arr = recvbuf.reshape(MPI.COMM_WORLD.Get_size(), 7, pool_size)
                results_processed_arr = np.transpose(results_processed_arr, (0, 2, 1)).reshape(-1, 7)
                results_processed = [dict(returns=r[0]) for r in results_processed_arr]
                returns = np.asarray([r['returns'] for r in results_processed])

                # ES update
                noise = noise[::1]
                returns = np.mean(returns.reshape(-1, 1), axis=1)
                phi_grad = relative_ranks(returns).dot(noise) / self.outer_n_worker - self.outer_l2 * phi
                phi -= adam.step(phi_grad)
                if self.save_model:
                    self.save_theta(phi, self.log.path_model)
        if self.save_model:
            self.log.dump_tabular()
            self.log.pickle_tf_vars()
        pool.close()

    def test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        return 100. * correct / len(self.test_loader.dataset)
