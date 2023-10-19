import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import calendar
import time
import config
import math
import os

def train_nn(nn, train_loader, valid_loader, lossfunction, optimizer, device='cpu', UUID='default'):
    print('Training on device', device)
    nn = nn.to(device)
    lossfunction = lossfunction.to(device)
    
    training_ID = int(calendar.timegm(time.gmtime()))
    if not UUID == 'default':
        UUID = f'{hash(UUID)}'
    print(f'The ID for this training is {UUID}_{training_ID}.')
    
    train_loss = []
    valid_loss = []
    best_valid_loss = math.inf
    patience = 0

    for epoch in range(10**10):
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            prediction_train = nn(x_train)
            L_train = lossfunction(prediction_train, y_train)
            train_loss.append(L_train.item())
            
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()
            
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                prediction_valid = nn(x_valid)
                L_valid = lossfunction(prediction_valid, y_valid)
                valid_loss.append(L_valid.item())

        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            torch.save(nn, f'./temp/NN_{UUID}_{training_ID}')
            patience = 0
        else:
            patience += 1

        if patience > 5000:
            print('Early stop.')
            break

        if not epoch % 500:
            print(f'| Epoch: {epoch:-8d} | Train loss: {L_train.item():.5f} | Valid loss: {L_valid.item():.5f} |')
    
    # remove temp files
    resulted_nn = torch.load(f'./temp/NN_{UUID}_{training_ID}')
    os.remove(f'./temp/NN_{UUID}_{training_ID}')
    
    print('Finished.')
    return resulted_nn, train_loss, valid_loss
