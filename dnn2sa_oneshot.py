########################################################################

'''

Author: Guo Jun-Lin (Kuo Chun-Lin)
Email : guojl19@mails.tsinghua.edu.cn
Date  : 2020.07.28 - 2021.04.04
'''

########################################################################

import os
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchrevision import CustomizedLinear

from cifar10 import CifarLoader

########################################################################


def gen_mask(row, col, percent=0.5, num_zeros=None):
    if num_zeros is None:
        # Total number being masked is 0.5 by default.
        num_zeros = int((row * col) * percent)

    mask = np.hstack([np.zeros(num_zeros),
                      np.ones(row * col - num_zeros)])
    np.random.shuffle(mask)
    return mask.reshape(row, col)


class Network(nn.Module):
    def __init__(self, in_size, out_size, ratio=[0, 0.5, 0]):
        super(Network, self).__init__()
        # self.fc1 = nn.Linear(in_size, 32)
        self.fc1 = CustomizedLinear(
            in_size, 32, mask=gen_mask(in_size, 32, ratio[0]))
        self.bn1 = nn.BatchNorm1d(32)
        # self.fc2 = nn.Linear(32, 16)
        self.fc2 = CustomizedLinear(32, 16, mask=gen_mask(32, 16, ratio[1]))
        self.bn2 = nn.BatchNorm1d(16)
        # self.fc3 = nn.Linear(16, out_size)
        self.fc3 = CustomizedLinear(
            16, out_size, mask=gen_mask(16, out_size, ratio[2]))
        self.bn3 = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()

        # Initialize parameters by following steps:
        # Link: https://pytorch.org/docs/stable/nn.init.html
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0, b=1)
                # nn.init.kaiming_normal_(m.weight,
                #                         mode='fan_out',
                #                         nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

########################################################################


class NetworkOptimization(object):
    """docstring for NetworkOptimization"""

    def __init__(self, batch_size, learning_rate, mask_ratio,
                 sa_cfg={'T': 5, 'k': 1, 'eta': 0.99},
                 data='mnist', satol=0.1, frac=0.7, edge_changed=1):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.sa = sa_cfg
        self.T = sa_cfg['T']
        self.satol = satol
        self.frac = frac
        self.edge_changed = edge_changed
        self.global_step = 0
        self.loss_prev = 1e+10  # A whatever big number.

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # --------------------------------------------------------------
        if data == 'mnist':
            train_dataset = torchvision.datasets.MNIST(
                root='./Datasets/', train=True,
                transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.MNIST(
                root='./Datasets/', train=False,
                transform=transforms.ToTensor(), download=True)
        elif data == 'fashion':
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./Datasets/', train=True,
                transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./Datasets/', train=False,
                transform=transforms.ToTensor(), download=True)
        elif data == 'cifar10':
            train_dataset = CifarLoader('cifar', 'data_batch',
                                        train=True, proc=False)
            test_dataset = CifarLoader('cifar', 'test_batch',
                                       train=False, proc=False)
        # --------------------------------------------------------------

        self.train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        self.test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=10000,
                                                shuffle=True)

        self.model = Network(in_size=3072, out_size=10, ratio=mask_ratio)
        self.model = self.model.to(device=self.device)
        # self.model_ckpt = copy.deepcopy(self.model)
        self.loss_ckpt = None

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss()

        # print(self.model.fc2.weight.data)
        self.total = len(train_dataset)

    def accuracy(self, outputs, labels):
        pred = torch.argmax(outputs, axis=1) == labels
        pred = pred.cpu().numpy()
        return np.sum(pred) / len(pred)

    def info(self, data, labels, epoch=0):
        # Test the trained model using testing set after each Epoch.
        outputs = self.model(data)
        losses = self.loss_func(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        print('[+] epoch: {0} | test acc: {1} | test loss: {2}'.format(
            epoch, np.round(accuracy, 3), np.round(losses.item(), 3)), end='\r')

        return losses.data.cpu().numpy(), accuracy

    def fit(self, epoch_num, save_step):
        test_data, test_label = next(iter(self.test))
        test_data = test_data.to(device=self.device)
        test_label = test_label.to(device=self.device)
        loss_rec, acc_rec, msk_rec = [], [], []
        epoch = 0

        msk = self.mask_ratio('fc2')

        for e in range(epoch_num):
            # iters = round(self.total / self.batch_size)
            # desc = '[+] Training {:>2}/{} Epochs'.format(e + 1, epoch)
            # for batch_img, batch_lab in tqdm(self.train, desc=desc,
            #                                  total=iters, unit=' batches'):

            for batch_img, batch_lab in self.train:
                self.global_step += 1
                batch_img = batch_img.to(device=self.device)
                batch_lab = batch_lab.to(device=self.device)

                outputs = self.model(batch_img.view(self.batch_size, -1))
                loss = self.loss_func(outputs, batch_lab)

                if self.global_step % save_step == 0:
                    acc = self.accuracy(outputs, batch_lab)
                    print('\n[-] loss: ', np.round(loss.item(), 3),
                          '| batch acc: ', np.round(acc, 3), end='\r')

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # Test the trained model using testing set after each Epoch.
            loss_cur, acc_cur = self.info(test_data.view(
                test_data.shape[0], -1), test_label, e + 1)
            loss_rec.append(loss_cur)
            acc_rec.append(acc_cur)
            msk_rec.append(np.round(msk, 3))

            # if np.abs(loss_cur - self.loss_prev) < 0.02:
            #     epoch = e + 1
            #     print('\nTraining process was stoped at {} Epoch'.format(epoch))
            #     break

            self.loss_prev = loss_cur
            print(acc_cur)

        return loss_rec, acc_rec, msk_rec, epoch_num - epoch

    # def link_reduce(self, model, layer, ratio):
    #     mask_ori = getattr(model, layer).mask.data.clone()
    #     # Find out the indices of all active connections.
    #     idx_T = torch.stack(torch.where(mask_ori == 1), axis=1)
    #     # Get the actual number of links we want to remove according to a ratio.
    #     wanted_links = int(len(mask_ori.flatten()) * ratio)
    #     # Randomly pick up some amount of links.
    #     idx_w = np.random.choice(
    #         np.arange(len(idx_T)), wanted_links, replace=False)
    #     # Remove the links by setting mask as "0".
    #     getattr(model, layer).mask.data[(idx_T[idx_w][:, 0],
    #                                      idx_T[idx_w][:, 1])] = 0
    #     return model

    def link_reduce(self, model, layer, ratio):
        weight_ori = getattr(model, layer).weight.data.clone()
        wanted_links = int(len(weight_ori.flatten()) * ratio)

        weight_flat = weight_ori.flatten()
        mask_flat = getattr(model, layer).mask.data.clone().flatten()

        # # Pick up the small values and prune them.  edge 1
        # idx = torch.argsort(torch.abs(weight_flat))

        # random pruning strategy.  edge 2
        idx = np.arange(len(weight_flat))
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)

        n = 0
        for i in idx:
            if n == wanted_links:
                break

            if mask_flat[i] == 0:
                continue

            mask_flat[i] = 0
            n += 1

        getattr(model, layer).mask.data = mask_flat.view(weight_ori.shape)
        return model

    def link_modify(self, model, layer, num_link):
        for _ in range(num_link):
            mask_ori = getattr(model, layer).mask.data.clone()

            # Find out the indices of all active connections.
            idx_T = torch.stack(torch.where(mask_ori == 1), axis=1)
            # Find out the indices of all masked connections.
            idx_F = torch.stack(torch.where(mask_ori == 0), axis=1)

            # Decide how many times of changes we want to go.
            # Randomly generate an integer to pick up 2 numbers.
            T = torch.randint(0, len(idx_T), (1,))[0]
            F = torch.randint(0, len(idx_F), (1,))[0]

            # Disconnect the picked connection ...
            mask_ori[idx_T[T][0], idx_T[T][1]] = 0
            # ... and connect the disconnected one.
            mask_ori[idx_F[F][0], idx_F[F][1]] = 1
            getattr(model, layer).mask.data = mask_ori

        return model

    def mask_ratio(self, layer):
        msk = getattr(self.model, layer).mask.data.cpu().numpy()
        return np.sum(msk) / len(msk.flatten())

    def FCSA(self, layer, reduce, metropolis):
        test_data, test_label = next(iter(self.test))
        test_data = test_data.to(device=self.device)
        test_label = test_label.to(device=self.device)

        loss_metro, acc_metro, update = 0, 0, 0
        self.sa['T'] = copy.deepcopy(self.T)

        self.model = self.link_reduce(self.model, layer, reduce)
        # self.model_ckpt = copy.deepcopy(self.model)
        msk = self.mask_ratio(layer)

        for i in np.arange(100):
            for m in range(metropolis):
                mask_ori = getattr(self.model, layer).mask.data.clone()
                self.model = self.link_modify(self.model, layer,
                                              self.edge_changed)

                outputs = self.model(test_data.view(test_data.shape[0], -1))
                loss_mdf = self.loss_func(outputs, test_label)
                # delta = np.abs(loss_mdf.data.cpu().numpy() - self.loss_prev)
                delta = loss_mdf.data.cpu().numpy() - self.loss_prev  # prev > curr

                rand = self.sa['T'] * self.sa['k'] * np.log(np.random.rand())
                if delta < -self.satol or rand < -delta:
                    # Accept the changes even though sometimes acc is worse.
                    # self.model_ckpt = copy.deepcopy(self.model)
                    self.loss_prev = loss_mdf.data.cpu().numpy()
                    update += 1
                else:
                    # Reject the changes and recover the model to original state.
                    getattr(self.model, layer).mask.data = mask_ori
                    # self.model = copy.deepcopy(self.model_ckpt)

                # print('>>> update >>> ', update)
                # Test the trained model using testing set.
                outputs = self.model(test_data.view(test_data.shape[0], -1))
                losses = self.loss_func(outputs, test_label)
                accuracy = self.accuracy(outputs, test_label)
                print('[+] round: {0} | metro: {1} | test acc: {2} | test loss: {3} | T: {4} | M: {5}%'.format(
                    i + 1, m + 1,
                    np.round(accuracy, 3),
                    np.round(losses.item(), 3),
                    np.round(self.sa['T'], 3),
                    np.round(msk * 100, 3)), end='\r')

                loss_metro = np.round(losses.item(), 5)
                acc_metro = np.round(accuracy, 5)

            # Temperature decrease at each epoch.
            self.sa['T'] *= self.sa['eta']

        outputs = self.model(test_data.view(test_data.shape[0], -1))
        losses = self.loss_func(outputs, test_label)
        accuracy = self.accuracy(outputs, test_label)

        loss_metro = np.round(losses.item(), 5)
        acc_metro = np.round(accuracy, 5)
        return loss_metro, acc_metro, msk


def save_csv(arrs, pth, epoch):
    df = pd.DataFrame(np.array(arrs), columns=range(1, epoch + 1))
    if os.path.exists(pth):
        arrs_old = pd.read_csv(pth, index_col=0).values.reshape(-1, epoch)
        arrs = np.concatenate((arrs_old, np.array(arrs)), axis=0)
        df = pd.DataFrame(arrs, columns=range(1, epoch + 1))

    df.to_csv(pth)


########################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30, help="Epoch size..")
    parser.add_argument('--batch', type=int, default=250, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning r.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat num.")
    parser.add_argument('--data', type=str, default='mnist',
                        help='select data.')
    parser.add_argument('--satol', type=float, default=0.0001,
                        help='SA tolerance for accepting new structure.')
    parser.add_argument('--frac', type=float, default=0.7,
                        help='fraction of batches.')
    parser.add_argument('--edge', type=int, default=10,
                        help='number of edge changes.')
    parser.add_argument('--temp', type=float, default=0.2, help="Temperature.")
    parser.add_argument('--k', type=float, default=1, help="k coefficient.")
    parser.add_argument('--eta', type=float, default=0.9, help="Decay coef.")
    parser.add_argument('--metro', type=str, default='1,5,10,20,50,100,200')
    parser.add_argument('--reduce', type=str, default='10,20,30,40,50,60,70,80,90,95,99,99.8')
    opt = parser.parse_args()

    EPOCH = opt.epoch
    BATCH = opt.batch
    LR = opt.lr
    REPEAT = opt.repeat
    DATA = opt.data
    SATOL = opt.satol
    FRAC = opt.frac
    EDGE = opt.edge
    CONFIG = {'T': opt.temp, 'k': opt.k, 'eta': opt.eta}
    METRO = np.array(opt.metro.split(',')).astype(int)
    REDUCE = np.array(opt.reduce.split(',')).astype(float) * 0.01

    # EPOCH = 30
    # BATCH = 125
    # LR = 0.001
    # REPEAT = 1
    # DATA = 'cifar10'
    # SATOL = 0.05
    # FRAC = 1.0
    # EDGE = 1
    # CONFIG = {'T': opt.temp, 'k': opt.k, 'eta': opt.eta}
    # METRO = [0]
    # REDUCE = [0.05]

    print('Experimental info:', {
          'epoch': opt.epoch, 'batch': opt.batch, 'lr': opt.lr,
          'repeat': opt.repeat, 'data': opt.data, 'satol': opt.satol,
          'frac': opt.frac, 'T': opt.temp, 'k': opt.k, 'eta': opt.eta,
          'reduce': REDUCE, 'metro': METRO})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[v] Using device: {}'.format(device))

    # Set up metropolis loop length. default='1,5,10,20,50,100,200'
    for m in METRO:
        folder = '{}_oneshot{}/metropolis_{}_rand'.format(DATA, EDGE, m)
        print('Save to: ', folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        for r in REDUCE:
            desc = '[+] Running {:>2}% Reduced Case'.format(r * 100)

            file_loss = 'loss_reduce{}.csv'.format(r * 100)
            file_acc = 'acc_reduce{}.csv'.format(r * 100)
            file_msk = 'msk_reduce{}.csv'.format(r * 100)

            for time in tqdm(range(REPEAT), desc=desc,
                             total=REPEAT, unit=' time'):

                if os.path.exists(os.path.join(folder, file_loss)):
                    arrs_old = pd.read_csv(os.path.join(
                        folder, file_loss), index_col=0).values
                    if len(arrs_old) >= REPEAT:
                        break

                nnOpt = copy.deepcopy(NetworkOptimization(batch_size=BATCH,
                                                          learning_rate=LR,
                                                          mask_ratio=[
                                                              0, 0.0, 0],
                                                          sa_cfg=CONFIG,
                                                          data=DATA,
                                                          satol=SATOL,
                                                          frac=FRAC,
                                                          edge_changed=EDGE))
                losses, accs, msks, epoch = nnOpt.fit(epoch_num=10,  # EPOCH
                                                      save_step=np.inf)
                np.savetxt(os.path.join(folder, 'w0_reduce{}.txt'.format(r * 100)),
                           nnOpt.model.fc2.weight.data.cpu().numpy())

                loss, acc, msk = nnOpt.FCSA('fc2', reduce=r, metropolis=m)
                losses[-1] = loss
                accs[-1] = acc
                msks[-1] = msk

                losses_new, accs_new, msks_new, epoch = nnOpt.fit(epoch_num=10,  # epoch
                                                                  save_step=np.inf)
                losses = losses + losses_new
                accs = accs + accs_new
                msks = msks + msks_new

                np.savetxt(os.path.join(folder, 'w_reduce{}.txt'.format(r * 100)),
                           nnOpt.model.fc2.weight.data.cpu().numpy())
                np.savetxt(os.path.join(folder, 'm_reduce{}.txt'.format(r * 100)),
                           nnOpt.model.fc2.mask.data.cpu().numpy())

                save_csv([losses], os.path.join(folder, file_loss),
                         epoch=len(losses))  # EPOCH
                save_csv([accs], os.path.join(folder, file_acc),
                         epoch=len(accs))       # EPOCH
                save_csv([msks], os.path.join(folder, file_msk),
                         epoch=len(msks))       # EPOCH
                del nnOpt

                print('[v] Finish {} time try under {} metro.'.format(time + 1, m))
                # break

            print('[v] Finish {:>2}% Reduced Case'.format(r * 100))
            # break
