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
        # define nonlinear circuits
        self.INV = INV
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = ACT.eta[2].detach().item() / \
            (1.-ACT.eta[2].detach().item()) * \
            (torch.sum(theta[:-2, :], axis=0)+theta[-1, :])
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
    def W(self):
        # to deal with case that the whole colume of theta is 0
        surrogate_theta = torch.ones_like(self.theta.abs())
        M, N = surrogate_theta.shape
        for n in range(N):
            if self.theta.abs()[:, n].sum() == 0.:
                surrogate_theta[:, n] = 0.
                surrogate_theta[:, n] = surrogate_theta[:, n].detach()
            else:
                surrogate_theta[:, n] = self.theta.abs()[:, n]

        W = surrogate_theta.abs() / torch.sum(surrogate_theta, axis=0, keepdim=True)
        W = torch.where(torch.isnan(W), torch.zeros_like(W), W)
        return W.to(self.device)

    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive
        a_extend = torch.cat([a,
                              torch.ones([a.shape[0], 1]).to(self.device),
                              torch.zeros([a.shape[0], 1]).to(self.device)], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:, -1] = torch.tensor(0.).to(self.device)

        z = torch.matmul(a_extend, self.W * positive) + \
            torch.matmul(a_neg, self.W * negative)
        return z

    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        self.mac_power = self.MAC_power(a_previous, z_new)
        return z_new

    @property
    def g_tilde(self):
        # scaled conductances
        g_initial = self.theta_.abs()
        g_min = g_initial.min(dim=0, keepdim=True)[0]
        scaler = self.args.pgmin / g_min
        return g_initial * scaler

    def MAC_power(self, x, y):
        x_extend = torch.cat([x,
                              torch.ones([x.shape[0], 1]).to(self.device),
                              torch.zeros([x.shape[0], 1]).to(self.device)], dim=1)
        x_neg = self.INV(x_extend)
        x_neg[:, -1] = 0.

        E = x_extend.shape[0]
        M = x_extend.shape[1]
        N = y.shape[1]

        positive = self.theta.clone().detach().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        Power = torch.tensor(0.).to(self.device)

        for m in range(M):
            for n in range(N):
                Power += self.g_tilde[m, n] * (
                    (x_extend[:, m]*positive[m, n]+x_neg[:, m]*negative[m, n])-y[:, n]).pow(2.).sum()
        Power = Power / E
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


class pSkipLayer(pLayer):
    def __init__(self, n_in, n_out, args, ACT, INV):
        super().__init__(n_in, n_out, args, ACT, INV)
        self.args = args
        # initialize conductances for weights
        theta = torch.rand([n_in, n_out])/100. + args.gmin
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    @property
    def theta(self):
        self.theta_.data.clamp_(0., self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    def MAC(self, a):
        return torch.matmul(a, self.W)

    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        self.mac_power = self.MAC_power(a_previous, z_new)
        return z_new

    @property
    def g_tilde(self):
        # scaled conductances
        g_initial = self.theta_.abs()
        g_min = g_initial.min(dim=0, keepdim=True)[0]
        scaler = self.args.pgmin / g_min
        return g_initial * scaler

    def MAC_power(self, x, y):
        E = x.shape[0]
        M = x.shape[1]
        N = y.shape[1]
        Power = torch.tensor(0.).to(self.device)
        for m in range(M):
            for n in range(N):
                Power += self.g_tilde[m, n] * ((x[:, m]-y[:, n]).pow(2.).sum())
        Power = Power / E
        return Power

    @property
    def soft_num_act(self):
        return torch.tensor(0.).to(self.device)

    @property
    def soft_num_neg(self):
        return torch.tensor(0.).to(self.device)

    def UpdateArgs(self, args):
        self.args = args


# ================================================================================================================================================
# ==============================================================  Printed Circuit  ===============================================================
# ================================================================================================================================================

class pNN(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()

        self.args = args

        # define nonlinear circuits
        self.act = TanhRT(args)
        self.inv = InvRT(args)

        # area
        self.area_theta = torch.tensor(args.area_theta).to(self.device)
        self.area_act = torch.tensor(args.area_act).to(self.device)
        self.area_neg = torch.tensor(args.area_neg).to(self.device)

        # connection and skip connection
        self.skipconnenction = args.skipconnection
        self.skips = [[0 for _ in range(len(topology))]
                      for _ in range(len(topology))]
        self.temp_skip_values = [
            [0 for _ in range(len(topology))] for _ in range(len(topology))]

        self.model = torch.nn.ModuleList()
        for i in range(len(topology)-1):
            self.model.append(
                pLayer(topology[i], topology[i+1], args, self.act, self.inv))
            if self.skipconnenction:
                for j in range(i+1, len(topology)):
                    # add skip connection from i to deeper layers
                    self.skips[i][j] = pSkipLayer(
                        topology[i], topology[j], args, self.act, self.inv)

    def forward(self, x):
        layers = len(self.model)
        # process input data layer by layer
        for i in range(layers):
            # normal layer: pass data from i to i+1 layer
            out = self.model[i](x)
            out = self.act(out)
            if self.skipconnenction:
                # skip connection: pass data from i to
                for j in range(i+1, layers):
                    self.temp_skip_values[i][j] = self.skips[i][j](x)

                # accumulate values from last and shallower layers
                for layer in range(i+1):
                    out = out + self.temp_skip_values[layer][i+1]
            x = out
        return out

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
        for i in range(len(self.skips)):
            for j in range(len(self.skips)):
                if hasattr(self.skips[i][j], 'soft_num_theta'):
                    soft_count += self.skips[i][j].soft_num_theta
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
        for i in self.skips:
            for j in i:
                if hasattr(j, 'UpdateArgs'):
                    j.UpdateArgs(args)


# ================================================================================================================================================
# =============================================================  pNN Loss function  ==============================================================
# ================================================================================================================================================

class pNNLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.area_baseline = torch.tensor(
            [542., 518., 643., 892., 625., 619., 533., 548., 885., 609., 614., 544., 577.]).to(args.DEVICE)

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

    def PowerEstimator(self, nn, x):
        _ = nn(x)
        return nn.Power

    def AreaEstimator(self, nn):
        return nn.Area

    def forward(self, nn, x, label):
        if self.args.powerestimator == 'power':
            return (1. - self.args.powerbalance) * self.standard(nn(x), label) + self.args.powerbalance * self.PowerEstimator(nn, x)
        elif self.args.areaestimator == 'area':
            return (1. - self.args.areabalance) * self.standard(nn(x), label) + self.args.areabalance * self.AreaEstimator(nn) / self.area_baseline[self.args.DATASET]
        else:
            return self.standard(nn(x), label)
