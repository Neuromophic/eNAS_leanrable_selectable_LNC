import numpy as np
import torch

# ================================================================================================================================================
# =====================================================  Learnable Negative Weight Circuit  ======================================================
# ================================================================================================================================================


class InvRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # R1n, k1, R3n, k2, R5n, Wn, k3
        # be careful, k1, k2, k3 are not normalized
        self.rt_ = torch.nn.Parameter(torch.tensor(
            [args.NEG_R1n, args.NEG_k1, args.NEG_R3n, args.NEG_k2, args.NEG_R5n, args.NEG_Wn, args.NEG_Ln]), requires_grad=True)
        # model
        package = torch.load('./utils/neg_model_package')
        self.eta_estimator = package['eta_estimator'].to(self.args.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.args.DEVICE)
        self.X_min = package['X_min'].to(self.args.DEVICE)
        self.Y_max = package['Y_max'].to(self.args.DEVICE)
        self.Y_min = package['Y_min'].to(self.args.DEVICE)
        # load power model
        package = torch.load('./utils/neg_power_model_package')
        self.power_estimator = package['power_estimator'].to(self.args.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.args.DEVICE)
        self.pow_X_min = package['X_min'].to(self.args.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.args.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.args.DEVICE)

    @property
    def RT_(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # calculate normalized (only R1n, R3n, R5n, Wn, Ln)
        RTn = torch.zeros([10]).to(self.args.DEVICE)
        RTn[0] = rt_temp[0]    # R1n
        RTn[2] = rt_temp[2]    # R3n
        RTn[4] = rt_temp[4]    # R5n
        RTn[5] = rt_temp[5]    # Wn
        RTn[6] = rt_temp[6]    # Ln
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # calculate R2, R4
        R2 = RT[0] * rt_temp[1]  # R2 = R1 * k1
        R4 = RT[2] * rt_temp[3]  # R4 = R3 * k2
        # stack new variable: R1, R2, R3, R4, R5, W, L
        RT_full = torch.stack([RT[0], R2, RT[2], R4, RT[4], RT[5], RT[6]])
        return RT_full

    @property
    def RT(self):
        # keep each component value in feasible range
        RT_full = torch.zeros([10]).to(self.args.DEVICE)
        RT_full[:7] = self.RT_.clone()
        RT_full[RT_full > self.X_max] = self.X_max[RT_full > self.X_max]    # clip
        RT_full[RT_full < self.X_min] = self.X_min[RT_full < self.X_min]    # clip
        return RT_full[:7].detach() + self.RT_ - self.RT_.detach()

    @property
    def RT_extend(self):
        # extend RT to 10 variables with k1 k2 and k3
        R1 = self.RT[0]
        R2 = self.RT[1]
        R3 = self.RT[2]
        R4 = self.RT[3]
        R5 = self.RT[4]
        W = self.RT[5]
        L = self.RT[6]
        k1 = R2 / R1
        k2 = R4 / R3
        k3 = L / W
        return torch.hstack([R1, R2, R3, R4, R5, W, L, k1, k2, k3])

    @property
    def RTn_extend(self):
        # normalize RT_extend
        return (self.RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power

    def forward(self, z):
        return - (self.eta[0] + self.eta[1] * torch.tanh((z - self.eta[2]) * self.eta[3]))


# ================================================================================================================================================
# ========================================================  Learnable Activation Circuit  ========================================================
# ================================================================================================================================================

class TanhRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # R1n, R2n, W1n, L1n, W2n, L2n
        self.rt_ = torch.nn.Parameter(
            torch.tensor([args.ACT_R1n, args.ACT_R2n, args.ACT_W1n, args.ACT_L1n, args.ACT_W2n, args.ACT_L2n]), requires_grad=True)

        # model
        package = torch.load('./utils/act_model_package')
        self.eta_estimator = package['eta_estimator'].to(self.args.DEVICE)
        self.eta_estimator.train(False)
        for n, p in self.eta_estimator.named_parameters():
            p.requires_grad = False
        self.X_max = package['X_max'].to(self.args.DEVICE)
        self.X_min = package['X_min'].to(self.args.DEVICE)
        self.Y_max = package['Y_max'].to(self.args.DEVICE)
        self.Y_min = package['Y_min'].to(self.args.DEVICE)
        # load power model
        package = torch.load('./utils/act_power_model_package')
        self.power_estimator = package['power_estimator'].to(self.args.DEVICE)
        self.power_estimator.train(False)
        for n, p in self.power_estimator.named_parameters():
            p.requires_grad = False
        self.pow_X_max = package['X_max'].to(self.args.DEVICE)
        self.pow_X_min = package['X_min'].to(self.args.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.args.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.args.DEVICE)

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # denormalization
        RTn = torch.zeros([9]).to(self.args.DEVICE)
        RTn[0] = rt_temp[0]    # R1n
        RTn[1] = rt_temp[1]    # R2n
        RTn[2] = rt_temp[2]    # W1n
        RTn[3] = rt_temp[3]    # L1n
        RTn[4] = rt_temp[4]    # W2n
        RTn[5] = rt_temp[5]    # L2n
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT[:6]

    @property
    def RT_extend(self):
        # extend RT to 9 variables with k1 k2 and k3
        R1 = self.RT[0]
        R2 = self.RT[1]
        W1 = self.RT[2]
        L1 = self.RT[3]
        W2 = self.RT[4]
        L2 = self.RT[5]
        k1 = R2 / R1
        k2 = L1 / W1
        k3 = L2 / W2
        return torch.hstack([R1, R2, W1, L1, W2, L2, k1, k2, k3])

    @property
    def RTn_extend(self):
        # normalize RT_extend
        return (self.RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.flatten()

    def forward(self, z):
        return self.eta[0] + self.eta[1] * torch.tanh((z - self.eta[2]) * self.eta[3])


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
