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

        self.best_train_genome = None
        self.best_val_genome = None

    def run(self, fitness_function, args, train_loader, valid_loader, setup):
        now = time.time()

        # initialization from previous training or create new
        if load_checkpoint(setup, args.temppath):
            self.generation, self.population, self.species, self.best_valid_genome = load_checkpoint(setup, args.temppath)
            self.msglogger.info(f'Restart previous training from {self.generation} generation.')
            print(f'Restart previous training from {self.generation} generation.')
        else:
            self.generation = 0
            patience = 0
            self.population = self.reproduction.create_new(self.config.genome_type, self.config.genome_config, self.config.pop_size, args)
            self.species = self.config.species_set_type(self.config.species_set_config)
            self.species.speciate(self.config, self.population, self.generation)

    
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
                save_checkpoint(generation, self.population, self.species, best_valid_genome, setup, args.temppath)
                patience = 0
            else:
                patience += 1

            msg = post_evaluate(self.population, best_train_genome, generation, self.config, 'Train')
            msg += post_evaluate(val_population, best_valid_genome, generation, self.config, 'Valid')

            if patience >= self.config.PATIENCE:
                self.msglogger.info('\nEarly-stopping based on validation loss.')
                print('\nEarly-stopping based on validation loss.')
                break

            self.population = self.reproduction.reproduce(self.config, self.population, self.species, self.config.pop_size, generation, args)

            msg += f'\nPatience for early-stopping: {patience}\nEpoch time: {time.time() - now:.3f}\n'
            now = time.time()
            
            if not generation % args.report_freq:
                self.msglogger.info(msg)
                print(msg)

            if not self.species.species:
                print('All species extinct. Create new population.')
                self.population = self.reproduction.create_new(self.config.genome_type, self.config.genome_config, self.config.pop_size, args)

            self.species.speciate(self.config, self.population, generation)

        return self.best_val_genome
