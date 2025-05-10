import os
import pandas as pd
import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from scipy.stats import norm
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from scipy import interpolate

sqrt_05 = np.sqrt(0.5)

pi_05 = 1 / np.sqrt(np.pi)
p4norm = sqrt_05 * pi_05

im_tfs = tfs.Compose([
  tfs.ToTensor(),
  tfs.Normalize([0.5], [0.5]) # 标准化
])

class FilteredDataset(Dataset):
    def __init__(self, data, labels, filter_condition):
        self.data = data
        self.labels = labels
        self.filtered_data = [(d, l) for d, l in zip(data, labels) if l in filter_condition]
    def __len__(self):
        return len(self.filtered_data)
    def __getitem__(self, idx):
        return self.filtered_data[idx]


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc_1 = nn.Linear(28 * 28, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_30 = nn.Linear(1024, 50) # mean
        self.fc_31 = nn.Linear(1024, 50) # var
        self.fc_4 = nn.Linear(50, 1024)
        self.fc_5 = nn.Linear(1024, 1024)
        self.fc_6 = nn.Linear(1024, 28 * 28)
    def encode(self, x):
        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        mu = self.fc_30(x)
        std = self.fc_31(x)
        std = std - torch.mean(std)
        std = torch.exp(std)
        #std = std / torch.mean(std)
        return mu, std
    def reparametrize(self, mu, std, eps):
        dich_r = torch.special.erf(sqrt_05 * mu / std) #(-1, 1)
        dich = 0.5 * dich_r # 0.5~-0.5
        z = eps * std + mu
        mask_r = z < 0
        mask = mask_r + 0.0
        mask = mask - 0.5 # 0.5 or -0.5
        dich_abs = torch.abs(dich - mask) #(0, 1)
        exp_in_r = torch.exp(-torch.pow(mu / std, 2) * 0.5)
        exp_in = std * p4norm * exp_in_r
        sub_mu =  mu * (0.5 - mask * dich_r) - mask * exp_in
        sub_mu = sub_mu / dich_abs
        mu_d = mu - sub_mu
        mu_d_2 = torch.pow(mu_d, 2)
        var = torch.pow(std, 2)
        sub_var =  (mu_d_2 + var) * (0.5 - mask * dich_r) - mask * (mu_d - sub_mu) * exp_in
        sub_std = torch.sqrt(0.5 * sub_var / dich_abs)
        eps = torch.FloatTensor(eps.size()).normal_()
        p_z = z
        z = eps * sub_std + sub_mu
        return z, dich_r, mask_r, sub_mu, sub_std, exp_in_r, p_z, dich_abs 
    def decode(self, z):
        z = self.fc_4(z)
        z = F.leaky_relu(z, 0.2)
        z = self.fc_5(z)
        z = F.leaky_relu(z, 0.2)
        z = self.fc_6(z)
        return z
    def forward(self, x, eps):
        mu_raw, std_raw = self.encode(x) # 编码
        z, dich, mask, mu, std, exp_in, p_z, dich_abs = self.reparametrize(mu_raw, std_raw, eps) # 重新参数化成正态分布
        return self.decode(z), dich, mu_raw, std_raw, mu, std, exp_in, z, p_z, dich_abs # 解码，同时输出均值方差
   

net = VAE() # 实例化网络
net.load_state_dict(torch.load('D:/exp/even/net_10000_97.51'))


def loss_function(mu, std):
    var = torch.pow(std, 2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return  KLD#25 * x.shape[0] * torch.log(mean) - torch.sum(torch.log(std))
   

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

train_set = MNIST('D:\mnist', train=True, transform=im_tfs,  download=True)
train_data = DataLoader(train_set, batch_size = 128, shuffle=True, drop_last=True) 
validate_set = MNIST('D:\mnist', train=False, transform=im_tfs,  download=True)
validate_data = DataLoader(validate_set, batch_size = 128, shuffle=True, drop_last=True) 
   
train_losses = []
MSE_losses = []
KLD_losses = []
dich_losses = []
mu_stds = []
std_meanes = []
dich_values = []
valids = []
_valids = []
for e in range(1):
    train_loss = 0
    MSE_loss = 0
    KLD_loss = 0
    dich_loss = 0
    mu_std = 0
    std_mean = 0
    dich_value = 0
    valid_value = 0
    _valid_value = 0
    for im, _ in validate_data:
        eps = torch.FloatTensor(128, 50).normal_()
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        recon_im, dich, mu_raw, std_raw, sub_mu, std, exp_in, z, p_z, dich_abs = net(im, eps)
        loss4e = dich_abs * 2
        mask = dich_abs > 1
        loss4e_raw = dich_abs * (mask + 0.0) + (~mask + 0.0)
        loss4e_raw = 0.5 * torch.log2(loss4e_raw) + 0.5
        loss4e_raw = torch.sum(loss4e_raw, axis = 1) + 1.0 #  * exp_in
        KLD = loss_function(sub_mu, std)
        eps = torch.FloatTensor(128, 50).normal_()
        gen_im, _dich, _mu_raw, _std_raw, _sub_mu, _std, _exp_in, _z, _p_z, _dich_abs = net(recon_im, eps)
        #valid_im = net.decode(torch.zeros(mu_raw.shape))
        valid_im = net.decode(2 * mu_raw)
        _valid_im = net.decode(2 * _mu_raw)
        mask = sub_mu * _sub_mu < 0#c2c
        mask = mask + 0.0
        _mask =  sub_mu * _mu_raw < 0
        _mask = _mask + 0.0
        mask = mask * _mask
        rate = torch.mean(mask + 0.0, axis = 1) #C2C
        mask = z * _z < 0
        mask = mask + 0.0
        mask = mask * _mask
        _mask =  z * _mu_raw < 0
        _mask = _mask + 0.0
        _rate = torch.mean(mask, axis = 1) #p2p
        _valid = torch.sum(1.25 * torch.pow(im - valid_im, 2) + 1.75 * torch.pow(im - _valid_im, 2), axis = 1)
        __valid = torch.sum(1.25 * torch.pow(valid_im - recon_im, 2) + 1.75 * torch.pow(_valid_im - recon_im, 2), axis = 1)
        _valid = torch.sqrt(_valid / 3)
        __valid = torch.sqrt(__valid / 3)
        ___valid = torch.sum(_valid * _rate) + torch.sum(((2 / 3 * _valid + 2 * __valid) * rate * loss4e_raw) / (loss4e_raw + 1/3))
        _valid_value = _valid_value + ___valid.detach().numpy()
    MSE_losses.append(MSE_loss / len(train_data))
    for im, _ in train_data:
        eps = torch.FloatTensor(128, 50).normal_()
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        recon_im, dich, mu_raw, std_raw, sub_mu, std, exp_in, z, p_z, dich_abs = net(im, eps)
        loss4e = dich_abs * 2
        mask = dich_abs > 1
        loss4e_raw = dich_abs * (mask + 0.0) + (~mask + 0.0)
        loss4e_raw = 0.5 * torch.log2(loss4e_raw) + 0.5
        loss4e_raw = torch.sum(loss4e_raw, axis = 1) + 1.0 #  * exp_in
        KLD = loss_function(sub_mu, std)
        eps = torch.FloatTensor(128, 50).normal_()
        gen_im, _dich, _mu_raw, _std_raw, _sub_mu, _std, _exp_in, _z, _p_z, _dich_abs = net(recon_im, eps)
        mean_im = net.decode(mu_raw)#mean but not saddle  * (mask + 0.0)
        #valid_im = net.decode(torch.zeros(mu_raw.shape))
        valid_im = net.decode(2 * mu_raw)
        _valid_im = net.decode(2 * _mu_raw)
        mask = sub_mu * _sub_mu < 0#c2c
        mask = mask + 0.0
        _mask =  sub_mu * _mu_raw < 0
        _mask = _mask + 0.0
        mask = mask * _mask
        rate = torch.mean(mask + 0.0, axis = 1) #C2C
        mask = z * _z < 0
        mask = mask + 0.0
        mask = mask * _mask
        _mask =  z * _mu_raw < 0
        _mask = _mask + 0.0
        _rate = torch.mean(mask, axis = 1) #p2p
        MSE = torch.sum(torch.pow(recon_im - im, 2), axis = 1) #generation - sample _recon_z z p2p
        _MSE = torch.sum(torch.pow(im - mean_im, 2), axis = 1)
        __MSE = torch.sum(torch.pow(mean_im - recon_im, 2), axis = 1) #mean - half mean (generation) _recon_z z C2C
        _valid = torch.sum(1.25 * torch.pow(im - valid_im, 2) + 1.75 * torch.pow(im - _valid_im, 2), axis = 1)
        __valid = torch.sum(1.25 * torch.pow(valid_im - recon_im, 2) + 1.75 * torch.pow(_valid_im - recon_im, 2), axis = 1)
        MSE = torch.sqrt(MSE)
        _MSE = torch.sqrt(_MSE)
        __MSE = torch.sqrt(__MSE)
        _valid = torch.sqrt(_valid / 3)
        __valid = torch.sqrt(__valid / 3)
        loss = torch.sum(_MSE * _rate) + torch.sum(((2 / 3 * _MSE + 2 * __MSE) * rate * loss4e_raw) / (loss4e_raw + 1/3))
        valid = torch.sum(_valid * _rate) + torch.sum(((2 / 3 * _valid + 2 * __valid) * rate * loss4e_raw) / (loss4e_raw + 1/3))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.detach().numpy()
        valid_value = valid_value + valid.detach().numpy()
        MSE_loss = MSE_loss + torch.sum(MSE).detach().numpy() / im.shape[0]
        KLD_loss = KLD_loss + KLD.detach().numpy() / im.shape[0]
        dich_loss = dich_loss + torch.sum(torch.abs(dich)).detach().numpy() / im.shape[0]
        dich_value = dich_value + torch.sum(dich).detach().numpy() / im.shape[0]
        mu_std = mu_std + (torch.std(mu_raw, unbiased = False)).detach().numpy()# / torch.mean(std)
        std_mean = std_mean + (torch.mean(std)).detach().numpy()# / torch.mean(std)
    _valids.append(_valid_value / len(validate_data))
    train_losses.append(train_loss / len(train_data))
    valids.append(valid_value / len(train_data))
    KLD_losses.append(KLD_loss / len(train_data))
    dich_losses.append(dich_loss / len(train_data))
    dich_values.append(dich_value / len(train_data))
    mu_stds.append(mu_std / len(train_data))
    std_meanes.append(std_mean / len(train_data))
    print('epoch: {}, Loss: {:.4f}'.format(e + 1, train_loss/len(train_data)))
    print('epoch: {}, MSE_loss: {:.4f}, KLD_loss: {:.4f}, valid: {:.4f}, _valid: {:.4f}, dich_loss: {:.4f}, dich_value: {:.4f}, mu_std: {:.4f}, std_mean: {:.4f}'.format(e + 1, MSE_loss / len(train_data), KLD_loss / len(train_data), valid_value / len(train_data), _valid_value / len(validate_data), dich_loss / len(train_data), dich_value / len(train_data), mu_std / len(train_data), std_mean / len(train_data)))

#torch.save(net.state_dict(), 'D:/exp/even/net_1000')
#torch.save(net.state_dict(), 'D:/exp/try/net_50_up')
name = ['train_loss', 'MSE_loss', 'KLD_loss', 'dich_loss', 'dich_value', 'mu_std', 'std_mean', 'valid_train', 'valid_test'] 
log = pd.DataFrame(columns = name)
log['train_loss'] = train_losses
log['MSE_loss'] = MSE_losses
log['KLD_loss'] = KLD_losses
log['dich_loss'] = dich_losses
log['dich_value'] = dich_values
log['mu_std'] = mu_stds
log['std_mean'] = std_meanes
log['valid_train'] = valids
log['valid_test'] = _valids
log.to_csv('D:/exp/even/log_10000.csv')
#log.to_csv('D:/exp/try/log_50_1_100_up.csv')
 


test_set = MNIST('D:\mnist', train=False, transform=im_tfs,  download=True)
test_data = DataLoader(test_set, batch_size = len(test_set), shuffle=False) # + 16 * e

#train_data = DataLoader(train_set, batch_size=60000, shuffle=True, drop_last=True)
im, _ = next(iter(test_data))
im = im.view(im.shape[0], -1)
im = Variable(im)
mu_raw, std_raw = net.encode(im) # 编码
#eps = torch.FloatTensor(10000, 50).normal_()
eps = torch.zeros(10000, 50)
#gen_im, dich, mu_raw, std_raw, mu, std, exp_in, z, p_z, dich_abs = net(im, eps)
z, dich, mask, sub_mu, sub_std, exp_in, p_z, dich_abs = net.reparametrize(mu_raw, std_raw, eps) # 重新参数化成正态分布
#gen = net.decode(sub_mu)
#mu_raw, std_raw = net.encode(gen)
#z, dich, mask, _sub_mu, sub_std, exp_in, p_z, dich_abs = net.reparametrize(mu_raw, std_raw, eps)
#gen = net.decode(sub_mu)
#_mu_raw, _std_raw = net.encode(gen)
#_z, _dich, _mask, _sub_mu, _sub_std, _exp_in, _p_z, _dich_abs = net.reparametrize(_mu_raw, _std_raw, eps)
#_gen_im, _dich, _mu_raw, _std_raw, _mu, _std, _exp_in, _z, _p_z, _dich_abs = net(gen_im, eps)
#z, dich, mask, _sub_mu, _sub_std, exp_in, sub_dich, loss4std = net.reparametrize(mu_raw, std_raw)
#z = net.reparametrize(mu, sig)
#eps = torch.FloatTensor(sub_std.size()).normal_()
#latents = (eps * sub_std + sub_mu)
#MSE = reconstruction_function(net.decode(latents), im) / im.shape[0]
#latents = im.detach().numpy() 
latents = sub_mu.detach().numpy() #0.5 * (sub_mu + _sub_mu)
labels = _.detach().numpy()
distances = np.zeros(len(latents))
counter = 0

for i in range(10000):
        for j in range(10000):
            distances[j] = pow(latents[i] - latents[j], 2).sum()
        distances[i] = distances.max()
        if(labels[np.argmin(distances)] == labels[i]):
            counter = counter + 1
   

counter




log = pd.read_csv('D:/exp/even/log_10000.csv')
valids = log['valid_train']
_valids = log['valid_test']
MSE_train_losses = log['MSE_train_loss']
KLD_train_losses = log['KLD_train_loss']
MSE_valid_losses = log['MSE_valid_loss'] 
KLD_valid_losses = log['KLD_valid_loss']
x = list(range(0, 87 + 1))
f = interpolate.interp1d(x , valids, kind = 'cubic')
_f = interpolate.interp1d(x , _valids, kind = 'cubic')
MSE_train_f = interpolate.interp1d(x , MSE_train_losses, kind = 'cubic')
KLD_train_f = interpolate.interp1d(x , KLD_train_losses, kind = 'cubic')
MSE_valid_f = interpolate.interp1d(x , MSE_valid_losses, kind = 'cubic')
KLD_valid_f = interpolate.interp1d(x , KLD_valid_losses, kind = 'cubic')
x = list(range(0, 87 * 6 + 1)) 
x = [i * (1 / 6) for i in x]
y = f(x)
_y = _f(x)
MSE_train = 4000 / 28  * MSE_train_f(x)
KLD_train = KLD_train_f(x)
MSE_valid = 4000 / 28  * MSE_valid_f(x)
KLD_valid = KLD_valid_f(x)
plt.plot(x, y, label='Convergence4Train', ls = '--', color = 'black')
plt.plot(x, _y, label='Convergence4Test', ls = '-', color = 'red')
plt.plot(x, MSE_train, label='4000×RMSE4Train', ls = '--', color = 'black')
plt.plot(x, MSE_valid, label='4000×RMSE4Test', ls = '-', color = 'green')
plt.plot(x, KLD_train, label='LKD4Train', ls = '--', color = 'black')
plt.plot(x, KLD_valid, label='LKD4Test', ls = '-', color = 'blue')
plt.xlabel('loop index of training 60000 samples',  fontdict={'family' : 'Times New Roman', 'size': 14})
plt.ylim(100, 2500)
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 14})
plt.show()

