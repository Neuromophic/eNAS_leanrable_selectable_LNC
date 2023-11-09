import numpy as np
import torch
import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(os.path.join(str(Path(os.getcwd()).parent), 'utils'))
import pNN
from feed_forward import *

def eval_genome(genome, config, loader, args):
    loss_fn = pNN.pNNLoss(args)

    genome_pruned = genome.get_pruned_copy(config.genome_config)
    
    num_node, num_edge = genome_pruned.size()
    neg_connections = [cg.key for cg in genome_pruned.connections.values() if PrintedNeuromorphicCircuit.theta_ste(cg.theta, config) < 0]

    neg_inputs = set()
    for conn_key in neg_connections:
        input_node, _ = conn_key
        neg_inputs.add(input_node)
    num_neg = len(neg_inputs)

    neg_biases = [1 for ng in genome.nodes.values() if PrintedNeuromorphicCircuit.theta_ste(ng.gb, config) < 0]
    num_neg =+ (len(neg_biases)>0)

    num_theta = num_edge + 2 * num_node
    num_act = num_node

    genome.num_theta, genome.num_act, genome.num_neg = num_theta, num_act, num_neg
    area = num_theta * args.area_theta + num_neg * args.area_neg + num_act * args.area_act

    net = PrintedNeuromorphicCircuit.create(genome_pruned, config, args)
    for inputs, target in loader:
        y_pred = net.forward(inputs, config)
        prediction = torch.tensor(np.array(y_pred)).permute(1, 2, 0)
        loss = loss_fn(prediction.detach(), target.detach()).numpy()

    acc = []
    for i in range(prediction.shape[0]):
        acc.append((prediction[i].detach().argmax(1) == target.detach()).float().mean().numpy())
    accuracy = numpy.mean(acc)
    std = numpy.std(acc)

    genome.fitness = accuracy - loss
    genome.area = area
    genome.accuracy = accuracy
    genome.std = std
        

def eval_genomes(population, config, loader, args):
    for _, genome in population:
        eval_genome(genome, config, loader, args)
        


def generation_report(population, species_set, generation):
    ng = len(population)
    ns = len(species_set.species)
    msg = f'\nPopulation of {ng} members in {ns} species' \
        + '\n ==================================================' \
        + '\n |  ID  | age | size |  fitness  | adj fit | stag |' \
        + '\n -------+-----+------+-----------+---------+-------'
    for sid in sorted(species_set.species):
        s = species_set.species[sid]
        age = generation - s.created
        n = len(s.members)
        f = "--" if s.fitness is None else f"{s.fitness:.3f}"
        af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
        st = generation - s.last_improved
        msg += f'\n | {sid:>4} | {age:>3} | {n:>4} | {f:>9} | {af:>7} | {st:>4} |'
    msg += '\n =================================================='
    return msg


def post_evaluate(population, best_genome, generation, config, stage):
    fitnesses = [c.fitness for c in population.values()]
    fit_mean = np.mean(fitnesses)
    genome = best_genome.get_pruned_copy(config.genome_config)
    
    msg = f'\n| {generation:-3d} | {stage} | Mean Fit: {fit_mean:.2e} | Best Fit: {best_genome.fitness:.2e} | #Node {genome.size()[0]:2d} | #Edge {genome.size()[1]:2d} | Accuracy: {best_genome.accuracy:.3f} | Std: {best_genome.std:.3e} |'
    return msg
