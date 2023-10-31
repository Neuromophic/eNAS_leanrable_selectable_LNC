class ConfigParameter(object):
    def __init__(self, name, value_type, default=None):
        self.name = name
        self.value_type = value_type
        self.default = default

    def interpret(self, param_dict):
        if self.default is not None:
            value = self.default
        else:
            try:
                value = param_dict[self.name]
            except KeyError:
                raise RuntimeError(f'Missing configuration item: {self.name}')
        return value


class DefaultClassConfig(object):
    def __init__(self, param_dict, param_list):
        self._params = param_list
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)


class Config(object):
    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('PATIENCE', int),
                ConfigParameter('GENERATION', int),]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, args):
        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        param = vars(args)
        for p in self.__params:
            setattr(self, p.name, param[p.name] if p.default is None else p.default)

        self.genome_config = genome_type.parse_config(param)
        self.species_set_config = species_set_type.parse_config(param)
        self.stagnation_config = stagnation_type.parse_config(param)
        self.reproduction_config = reproduction_type.parse_config(param)
