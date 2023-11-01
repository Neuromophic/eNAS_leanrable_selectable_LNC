import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'NEAT_LNC_genome'))
import torch
import NEAT_LNC_node as neat
import pprint
from utils import *
from configuration import *
from NEAT_args import *

args = parser.parse_args()
EA_args = EA_parser.parse_args([])


args = FormulateArgs(args)

train_loader, datainfo = GetDataLoader(args, 'train')
valid_loader, datainfo = GetDataLoader(args, 'valid')
test_loader, datainfo = GetDataLoader(args, 'test')
pprint.pprint(datainfo)

x_train, y_train = train_loader.to_list()
x_valid, y_valid = valid_loader.to_list()
x_test, y_test = test_loader.to_list()

SetSeed(args.SEED)

setup = f"NEAT_data_{datainfo['dataname']}_seed_{args.SEED:02d}_epsilon_{args.e_train}"
print(f'Training setup: {setup}.')
MakeFolder(args)

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

    SetSeed(args.SEED)

    config = neat.Config(neat.PNCGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, EA_args)
    p = neat.Population(config, msglogger, args)
    winner = p.run(neat.eval_genomes, args, train_loader, valid_loader)

    winner_pruned = winner.get_pruned_copy(config.genome_config)
    torch.save(winner_pruned, f'{args.savepath}/{setup}.model')

    msglogger.info('Training if finished.')
    print('Training if finished.')

    