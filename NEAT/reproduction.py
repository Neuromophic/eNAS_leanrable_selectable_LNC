import math
import random
from itertools import count
import numpy as np
from config import ConfigParameter, DefaultClassConfig
from utility import *


class DefaultReproduction(DefaultClassConfig):

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('genome_elitism', int),
                                   ConfigParameter('survival_threshold', float)])

    def __init__(self, config, msglogger, stagnation):
        self.reproduction_config = config
        self.msglogger = msglogger
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):

        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            diff = (s - ps) * 0.5
            delta = int(round(diff))
            spawn = ps
            if abs(delta) > 0:
                spawn += delta
            spawn_amounts.append(spawn)

        # Normalize the spawn amounts
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, config, old_population, species, pop_size, generation):
        all_fitnesses = []
        remaining_species = []
        msg_stag = ''
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                msg_stag += f"\nSpecies {stag_sid} with {len(stag_s.members)} members is stagnated: removing it"
            else:
                all_fitnesses.extend(
                    m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        if not remaining_species:
            species.species = {}
            return {}

        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        for afs in remaining_species:
            # adjusted fitness
            msf = np.mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / (max_fitness - min_fitness + 1e-10)
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]

        msg = generation_report(old_population, species, generation) + msg_stag
        self.msglogger.info(msg)
        print(msg, end='')

        # number of new members for each species in the new generation
        previous_sizes = [len(s.members) for s in remaining_species]
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, self.reproduction_config.genome_elitism)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            old_members.sort(reverse=True, key=lambda x: x[1].fitness)
            for i, m in old_members[:self.reproduction_config.genome_elitism]:
                new_population[i] = m
                spawn -= 1

            if spawn <= 0:
                continue

            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold * len(old_members)))
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.crossover(parent1, parent2)
                child.mutate(config.genome_config)

                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
