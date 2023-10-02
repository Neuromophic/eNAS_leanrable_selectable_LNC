#!/usr/bin/env python
#SBATCH --job-name=AreaAware
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
sys.path.append(os.path.join(os.getcwd(), 'NEAT'))
import torch
import NEAT as neat
import pprint
from utils import *
from configuration import *
from NEAT_args import *

args = parser.parse_args()
EA_args = EA_parser.parse_args()

reference_areas = {'acuteinflammation': 542, 'balancescale': 518,'breastcancerwisc': 643,'cardiotocography3clases': 892,
                   'energyy1': 625, 'energyy2': 619, 'iris': 533, 'mammographic': 548, 'pendigits':885,'seeds': 609,
                   'tictactoe': 614, 'vertebralcolumn2clases': 544, 'vertebralcolumn3clases': 577}
reference_losses = {'acuteinflammation': 0.05, 'balancescale': 0.05,'breastcancerwisc': 0.05,'cardiotocography3clases': 0.1,
                   'energyy1': 0.1, 'energyy2': 0.1, 'iris': 0.05, 'mammographic': 0.25, 'pendigits':0.05,'seeds': 0.1,
                   'tictactoe': 0.01, 'vertebralcolumn2clases': 0.25, 'vertebralcolumn3clases': 0.2}

for seed in range(10):
    args.SEED = seed
    args = FormulateArgs(args)
    
    train_loader, datainfo = GetDataLoader(args, 'train')
    valid_loader, datainfo = GetDataLoader(args, 'valid')
    test_loader, datainfo = GetDataLoader(args, 'test')
    pprint.pprint(datainfo)

    x_train, y_train = train_loader.to_list()
    x_valid, y_valid = valid_loader.to_list()
    x_test, y_test = test_loader.to_list()

    SetSeed(args.SEED)

    setup = f"NEAT_data_{datainfo['dataname']}_seed_{args.SEED:02d}_Penalty_{args.areaestimator}_Factor_{args.areabalance:.5f}"
    print(f'Training setup: {setup}.')

    msglogger = GetMessageLogger(args, setup, EA_args)
    msglogger.info(f'Training network on device: {args.DEVICE}.')
    msglogger.info(f'Training setup: {setup}.')
    msglogger.info(datainfo)

    EA_args.num_inputs = datainfo['N_feature']
    EA_args.num_outputs = datainfo['N_class']

    if os.path.isfile(f'{args.savepath}/{setup}.model'):
        print(f'{setup} exists, skip this training.')
        msglogger.info('Training was already finished.')
    else:

        
        args.area_reference = reference_areas[datainfo["dataname"]]
        args.loss_reference = reference_areas[datainfo["dataname"]]

        SetSeed(args.SEED)

        config = neat.Config(neat.PNCGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, EA_args)
        p = neat.Population(config, msglogger, args)
        winner = p.run(neat.eval_genomes, train_loader, valid_loader)

        winner_pruned = winner.get_pruned_copy(config.genome_config)
        torch.save(winner_pruned, f'{args.savepath}/{setup}.model')

        msglogger.info('Training if finished.')
        print('Training if finished.')

        