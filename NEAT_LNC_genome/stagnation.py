import sys
from config import ConfigParameter, DefaultClassConfig


class DefaultStagnation(DefaultClassConfig):
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('max_stagnation', int),
                                   ConfigParameter('species_elitism', int)])

    def __init__(self, config):
        self.stagnation_config = config

    def update(self, species_set, generation):
        species_data = []
        for sid, s in species_set.species.items():
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            fitness_values = list(s.get_fitnesses())
            s.fitness = sum(map(float, fitness_values)) / len(fitness_values)
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation

            if (len(species_data) - idx) <= self.stagnation_config.species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result
    