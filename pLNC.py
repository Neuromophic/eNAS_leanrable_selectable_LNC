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
# =====================================================  Learnable Negative Weight Circuit  ======================================================
# ================================================================================================================================================

class InvRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # R1, R2, R3, W1, L1, W2, L2, W3, L3
        self.rt_ = torch.nn.Parameter(torch.tensor(
            [args.NEG_R1n, args.NEG_k1, args.NEG_R3n, args.NEG_k2, args.NEG_R5n, args.NEG_Wn, args.NEG_Ln]), requires_grad=True) # need to be changed
        # model
        package = torch.load('./utils/neg_model_package')                                                                        # need to be changed
        self.eta_estimator = package['eta_estimator'].to(self.args.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.args.DEVICE)
        self.X_min = package['X_min'].to(self.args.DEVICE)
        self.Y_max = package['Y_max'].to(self.args.DEVICE)
        self.Y_min = package['Y_min'].to(self.args.DEVICE)
        # load power model
        package = torch.load('./utils/neg_power_model_package')                                                                  # need to be changed
        self.power_estimator = package['power_estimator'].to(self.args.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.args.DEVICE)
        self.pow_X_min = package['X_min'].to(self.args.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.args.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.args.DEVICE)

    @property
    def RTn_extend(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([12]).to(self.args.DEVICE)
        RTn[:9] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # calculate ratios
        RT[9]  = RT[3] / RT[4]  # k1 = W1 / L1
        RT[10] = RT[5] / RT[6]  # k2 = W2 / L2
        RT[11] = RT[7] / RT[8]  # k3 = W3 / L3
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
    def RTn_extend(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # denormalization
        RTn = torch.zeros([9]).to(self.args.DEVICE)
        RTn[:6] = rt_temp
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # with ratio
        RT_extend = torch.stack([RT[0], RT[1], RT[2], RT[3], RT[4], RT[5], RT[1]/RT[0], RT[3]/RT[2], RT[5]/RT[4]])
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

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
    

