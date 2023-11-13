import numpy as np
import torch
from pLNC import *

# ================================================================================================================================================
# ===============================================================  Printed Layer  ================================================================
# ================================================================================================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, ACT, INV):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # define nonlinear circuits
        self.ACT = ACT
        self.INV = INV
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin
        
        eta_2 = ACT.eta.mean(0)[2].detach().item()
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = eta_2 / (1.-eta_2) * (torch.sum(theta[:-2, :], axis=0)+theta[-1, :])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def theta_noisy(self):
        mean = self.theta.repeat(self.N, 1, 1)
        nosie = ((torch.rand(mean.shape) * 2.) - 1.) * self.epsilon + 1.
        return mean * nosie 

    @property
    def W(self):
        # to deal with case that the whole colume of theta is 0
        G = torch.sum(self.theta_noisy.abs(), axis=1, keepdim=True)
        W = self.theta_noisy.abs() / (G + 1e-10)
        return W.to(self.device)

    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta_noisy.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        ones_tensor = torch.ones([a.shape[0], a.shape[1], 1]).to(self.device)
        zeros_tensor = torch.zeros_like(ones_tensor).to(self.device)
        a_extend = torch.cat([a, ones_tensor, zeros_tensor], dim=2)

        a_neg = self.INV(a_extend)
        a_neg[:, :, -1] = torch.tensor(0.).to(self.device)

        z = torch.matmul(a_extend, self.W * positive) + torch.matmul(a_neg, self.W * negative)

        return z

    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        self.mac_power = self.MAC_power(a_previous, z_new)
        a_new = self.ACT(z_new)
        return a_new

    @property
    def g_tilde(self):
        # scaled conductances
        g_initial = self.theta_.abs()
        g_min = g_initial.min(dim=0, keepdim=True)[0]
        scaler = self.args.pgmin / g_min
        return g_initial * scaler

    def MAC_power(self, x, y):
        x_extend = torch.cat([x,
                              torch.ones([x.shape[0], x.shape[1], 1]).to(self.device),
                              torch.zeros([x.shape[0], x.shape[1], 1]).to(self.device)], dim=2)
        x_neg = self.INV(x_extend)
        x_neg[:, :, -1] = 0.

        V = x_extend.shape[0]
        E = x_extend.shape[1]
        M = x_extend.shape[2]
        N = y.shape[2]

        positive = self.theta_noisy.clone().detach().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        Power = torch.tensor(0.).to(self.device)

        for v in range(V):
            for m in range(M):
                for n in range(N):
                    Power += self.g_tilde[m, n] * (
                        (x_extend[v, :, m]*positive[v, m, n]+x_neg[v, :, m]*negative[v, m, n])-y[v, :, n]).pow(2.).sum()
        Power = Power / E / V
        return Power

    @property
    def soft_num_theta(self):
        # forward pass: number of theta
        nonzero = self.theta.clone().detach().abs()
        nonzero[nonzero > 0] = 1.
        N_theta = nonzero.sum()
        # backward pass: pvalue of the minimal negative weights
        soft_count = torch.sigmoid(self.theta.abs())
        soft_count = soft_count * nonzero
        soft_count = soft_count.sum()
        return N_theta.detach() + soft_count - soft_count.detach()

    @property
    def soft_num_act(self):
        # forward pass: number of act
        nonzero = self.theta.clone().detach().abs()[:-2, :]
        nonzero[nonzero > 0] = 1.
        N_act = nonzero.max(0)[0].sum()
        # backward pass: pvalue of the minimal negative weights
        soft_count = torch.sigmoid(self.theta.abs()[:-2, :])
        soft_count = soft_count * nonzero
        soft_count = soft_count.max(0)[0].sum()
        return N_act.detach() + soft_count - soft_count.detach()

    @property
    def soft_num_neg(self):
        # forward pass: number of negative weights
        positive = self.theta.clone().detach()[:-2, :]
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive
        N_neg = negative.max(1)[0].sum()
        # backward pass: pvalue of the minimal negative weights
        soft_count = 1 - torch.sigmoid(self.theta[:-2, :])
        soft_count = soft_count * negative
        soft_count = soft_count.max(1)[0].sum()
        return N_neg.detach() + soft_count - soft_count.detach()

    def UpdateArgs(self, args):
        self.args = args
    
    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        self.INV.N = N
        self.INV.epsilon = epsilon


# ================================================================================================================================================
# ==============================================================  Printed Circuit  ===============================================================
# ================================================================================================================================================

class pNN(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()

        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train

        # define nonlinear circuits
        self.act = TanhRT(args)
        self.inv = InvRT(args)

        # area
        self.area_theta = torch.tensor(args.area_theta).to(self.device)
        self.area_act = torch.tensor(args.area_act).to(self.device)
        self.area_neg = torch.tensor(args.area_neg).to(self.device)

        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module(f'{i}-th Layer', pLayer(topology[i], topology[i+1], args, self.act, self.inv))

    def forward(self, x):
        x = x.repeat(self.N, 1, 1)
        return self.model(x)

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def soft_count_neg(self):
        soft_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'soft_num_neg'):
                soft_count += l.soft_num_neg
        return soft_count

    @property
    def soft_count_act(self):
        soft_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'soft_num_act'):
                soft_count += l.soft_num_act
        return soft_count

    @property
    def soft_count_theta(self):
        soft_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'soft_num_theta'):
                soft_count += l.soft_num_theta
        return soft_count

    @property
    def power_neg(self):
        return self.inv.power * self.soft_count_neg

    @property
    def power_act(self):
        return self.act.power * self.soft_count_act

    @property
    def power_mac(self):
        power_mac = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'mac_power'):
                power_mac += l.mac_power
        return power_mac

    @property
    def Power(self):
        return self.power_neg + self.power_act + self.power_mac

    @property
    def Area(self):
        return self.area_neg * self.soft_count_neg + self.area_act * self.soft_count_act + self.area_theta * self.soft_count_theta

    def GetParam(self):
        weights = [p for name, p in self.named_parameters()
                   if name.endswith('.theta_')]
        nonlinear = [p for name, p in self.named_parameters()
                     if name.endswith('.rt_')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights

    def UpdateArgs(self, args):
        self.args = args
        self.act.args = args
        self.inv.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)
    
    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        self.act.N = N
        self.act.epsilon = epsilon
        self.inv.N = N
        self.inv.epsilon = epsilon
        for layer in self.model:
            if hasattr(layer, 'UpdateVariation'):
                layer.UpdateVariation(N, epsilon)


# ================================================================================================================================================
# =============================================================  pNN Loss function  ==============================================================
# ================================================================================================================================================

class pNNLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L
    
    def CELoss(self, prediction, label):
        fn = torch.nn.CrossEntropyLoss()
        return fn(prediction, label)

    def forward(self, y, label):
        N = y.shape[0]
        loss = torch.tensor(0.).to(self.args.DEVICE)
        if self.args.metric == 'acc':
            for n in range(N):
                loss += self.CELoss(y[n,:,:], label)
        elif self.args.metric == 'maa':
            for n in range(N):
                loss += self.standard(y[n,:,:], label)
        
        return loss / N


