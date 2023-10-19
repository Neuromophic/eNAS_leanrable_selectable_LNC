#!/usr/bin/env python

#SBATCH --job-name=ReLUPsim

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import sys
import os
sys.path.append(os.getcwd())
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import training
import config
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = int(sys.argv[1])

for num_layer in range(3,11):
    for lr in range(-3,-6,-1):

        exp_setup = f'{num_layer}_{lr}_{seed}'
        print(f'The experiment setup is {exp_setup}.')
        
        if os.path.exists(f'./NNs/ReLUP_{exp_setup}.model'):
            pass
        else:
            
            a = torch.load('./data/pReLU_power.ds')
        
            X, Y = a['X'], a['Y']
            Xn, Yn = a['Xn'], a['Yn']
            X_min, X_max = a['X_min'], a['X_max']
            Y_min, Y_max = a['Y_min'], a['Y_max']
        
            X_learn, Y_learn = a['X_learn'], a['Y_learn']
            X_train, Y_train = a['X_train'], a['Y_train']
            X_valid, Y_valid = a['X_valid'], a['Y_valid']
            X_test , Y_test  = a['X_test'] , a['Y_test']
        
            Xn_learn, Yn_learn = a['Xn_learn'].to(device), a['Yn_learn'].to(device)
            Xn_train, Yn_train = a['Xn_train'].to(device), a['Yn_train'].to(device)
            Xn_valid, Yn_valid = a['Xn_valid'].to(device), a['Yn_valid'].to(device)
            Xn_test , Yn_test  = a['Xn_test'].to(device) , a['Yn_test'].to(device)
        
        
            train_data = TensorDataset(Xn_train, Yn_train)
            valid_data = TensorDataset(Xn_valid, Yn_valid)
            test_data  = TensorDataset(Xn_test, Yn_test)
        
            train_loader = DataLoader(train_data, batch_size=len(train_data))
            valid_loader = DataLoader(valid_data, batch_size=len(valid_data))
            test_loader  = DataLoader(test_data, batch_size=len(test_data))
        
            # topology = (np.round(np.logspace(np.log(X.shape[1]),
            #                                     np.log(Y.shape[1]),
            #                                     num=num_layer, base=np.e))).astype(int)
            topology = [X.shape[1]] + [10 for _ in range(num_layer-1)] + [Y.shape[1]]
        
            config.SetSeed(seed)
            model = torch.nn.Sequential().to(device)
            for t in range(len(topology)-1):
                model.add_module(f'{t}-MAC', torch.nn.Linear(topology[t], topology[t+1]))
                model.add_module(f'{t}-ACT', torch.nn.PReLU())
        
            lossfunction = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=10**lr)
        
            model, train_loss, valid_loss = training.train_nn(model, train_loader, valid_loader, lossfunction, optimizer, device, UUID=exp_setup)
            torch.save(model, f'./NNs/ReLUP_{exp_setup}.model')
            
            plt.figure()
            plt.plot(train_loss, label='train')
            plt.plot(valid_loss, label='valid')
            plt.savefig(f'./NNs/train_curve_{exp_setup}.pdf', format='pdf', bbox_inches='tight')
