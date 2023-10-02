import time
import copy
from utility import *

class Population(object):
    def __init__(self, config, msglogger, args, initial_state=None):
        self.msglogger = msglogger
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config)
        self.reproduction = config.reproduction_type(config.reproduction_config, self.msglogger, stagnation)
        self.args = args

        if initial_state is None:
            self.population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
            self.species = config.species_set_type(config.species_set_config)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_train_genome = None
        self.best_val_genome = None

    def run(self, fitness_function, train_loader, valid_loader):
        now = time.time()
        patience = 0
        for generation in range(self.generation, self.config.GENERATION):
            val_population = copy.deepcopy(self.population)

            fitness_function(list(self.population.items()), self.config, loader=train_loader, args=self.args)
            fitness_function(list(val_population.items()), self.config, loader=valid_loader, args=self.args)

            best_train_genome = None
            for train_genome in self.population.values():
                if best_train_genome is None or train_genome.fitness > best_train_genome.fitness:
                    best_train_genome = train_genome
            best_valid_genome = None
            for val_genome in val_population.values():
                if best_valid_genome is None or val_genome.fitness > best_valid_genome.fitness:
                    best_valid_genome = val_genome

            if self.best_val_genome is None or best_valid_genome.fitness > self.best_val_genome.fitness:
                self.best_val_genome = best_valid_genome
                patience = 0
            else:
                patience += 1

            msg = post_evaluate(self.population, best_train_genome, generation, self.config, 'Train')
            msg += post_evaluate(val_population, best_valid_genome, generation, self.config, 'Valid')

            if patience >= self.config.PATIENCE:
                self.msglogger.info('\nEarly-stopping based on validation loss.')
                print('\nEarly-stopping based on validation loss.')
                break

            self.population = self.reproduction.reproduce(self.config, self.population, self.species, self.config.pop_size, generation)

            msg += f'\nPatience for early-stopping: {patience}\nEpoch time: {time.time() - now:.3f}\n'
            now = time.time()
            self.msglogger.info(msg)
            print(msg)

            if not self.species.species:
                print('All species extinct. Create new population.')
                self.population = self.reproduction.create_new(self.config.genome_type, self.config.genome_config, self.config.pop_size)

            self.species.speciate(self.config, self.population, generation)

        return self.best_val_genome
