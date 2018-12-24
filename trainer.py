import torch
from tool import logz
import numpy as np
from network import loss_net
from network import CNN
from mpi4py import MPI
import time
import torch.optim as optim
import torch.nn.functional as F
from tool.utils import PiecewiseSchedule, Adam, relative_ranks


class Trainer(object):
    def __init__(self, target_model, device, train_dataset, test_dataset, test_batch_size, logdir,
                 inner_epoch_freq, inner_batch_size, inner_lr, inner_momentum,
                 outer_n_worker, outer_lr, outer_n_epoch, outer_std, outer_l2):
        self.target_model = target_model
        self.device = device
        self.test_batch_size = test_batch_size
        self.logdir = logdir
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=inner_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

        self.inner_epoch_freq = inner_epoch_freq
        self.inner_batch_size = inner_batch_size
        self.inner_lr = inner_lr
        self.inner_momentum = inner_momentum

        self.outer_n_worker = outer_n_worker
        self.outer_lr = outer_lr
        self.outer_n_epoch = outer_n_epoch
        self.outer_std = outer_std
        self.outer_l2 = outer_l2

    def train(self, n_cpu):
        logz.configure_output_dir(self.logdir)
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nodes=n_cpu)
        outer_lr_scheduler = PiecewiseSchedule([(0, self.outer_lr),
                                                (int(self.outer_n_epoch / 2), self.outer_lr * 0.1)],
                                               outside_value=self.outer_lr * 0.1)
        evoluted_loss = loss_net()
        phi = evoluted_loss.parameters()
        num_params = len(phi)
        adam = Adam(shape=(num_params, ), beta1=0., stepsize=self.outer_lr, dtype=np.float32)

        def inner_train(phi, pool_rank):
            model = CNN(input_channel=1).to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=self.inner_lr, momentum=self.inner_momentum)
            model.train()
            eloss = loss_net()
            eloss.set_params_1d(phi)
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                target_output = self.target_model(data)
                soft_target = eloss.loss(torch.cat((target, target_output)))
                loss = F.nll_loss(output, soft_target)
                loss.backward()
                optimizer.step()
            accuracy = self.test(model)
            print('Worker %d, test accuracy: %f' % (pool_rank, accuracy))
            return dict(accuracy=accuracy)

        for epoch in range(self.outer_n_epoch):
            adam.stepsize = outer_lr_scheduler.value(epoch)

            noise = np.random.randn(self.outer_n_worker, num_params)
            phi_noise = phi[np.newaxis, :] + noise * self.outer_std
            phi_noise = phi_noise.reshape(MPI.COMM_WORLD.Get_size(), -1)

            recvbuf = np.empty(phi_noise.shape[1], dtype='float')
            MPI.COMM_WORLD.Scatter(phi_noise, recvbuf, int_root=0)
            phi_noise = recvbuf.reshape(-1, num_params)

            start_time = time.time()
            pool_size = int(self.outer_n_worker / MPI.COMM_WORLD.Get_size())
            results = pool.amap(inner_train, phi_noise, range(pool_size)).get()

            result = [np.mean(r['accuracy']) for r in results]

            recvbuf = np.empty([MPI.COMM_WORLD.Get_size(), 7 * pool_size],
                               # 7 = number of scalars in results vector
                               dtype='float') if MPI.COMM_WORLD.Get_rank() == 0 else None
            results_processed_arr = np.asarray([result], dtype='float').ravel()
            MPI.COMM_WORLD.Gather(results_processed_arr, recvbuf, root=0)

            # outer loop  update at node 0
            if MPI.COMM_WORLD.Get_rank() == 0:
                end_time = time.time()
                print('All inner loops completed, returns gathered ({:.2f} sec).'.format(time.time() - start_time))
                results_processed_arr = recvbuf.reshape(MPI.COMM_WORLD.Get_size(), 7, pool_size)
                results_processed_arr = np.transpose(results_processed_arr, (0, 2, 1)).reshape(-1, 7)
                results_processed = [dict(returns=r[0],
                                          update_time=r[1],
                                          env_time=r[2],
                                          ep_length=r[3],
                                          n_ep=r[4],
                                          mean_ep_kl=r[5],
                                          final_rets=r[6]) for r in results_processed_arr]
                returns = np.asarray([r['returns'] for r in results_processed])

                # ES update
                noise = noise[::1]
                returns = np.mean(returns.reshape(-1, 1), axis=1)
                phi_grad = relative_ranks(returns).dot(noise) / self.outer_n_worker - self.outer_l2 * phi
                phi -= adam.step(phi_grad)

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
        return correct
