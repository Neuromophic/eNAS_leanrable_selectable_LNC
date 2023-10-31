from random import gauss, random, uniform, choice
from config import ConfigParameter


class BaseAttribute(object):
    def __init__(self, name):
        self.name = name
        for n in self._config_items:
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name):
        return f"{self.name}_{config_item_base_name}"

    def get_config_params(self):
        return [ConfigParameter(self.config_item_name(n), ci[0], ci[1])
                for n, ci in self._config_items.items()]


class FloatAttribute(BaseAttribute):
    _config_items = {"init_type": [str, None],
                     "mutate_rate": [float, 0.5],
                     "mutate_power": [float, 0.5]}

    def init_value(self, config, args):
        init_type = getattr(config, self.init_type_name)
        if 'g_init' in init_type:
            return uniform(0., 0.1)
        elif 'gb_init' in init_type:
            return uniform(0., 0.1)
        elif 'gd_init' in init_type:
            return config.gmax - uniform(0., 0.1)
        else:
            try:
                return getattr(args, init_type)
            except ValueError:
                raise ValueError(f'Unknown init_type: {init_type}')

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)
        mutate_power = getattr(config, self.mutate_power_name)

        if random() < mutate_rate:
            return value + gauss(0., mutate_power)
        else:
            return value


class BoolAttribute(BaseAttribute):
    _config_items = {"mutate_rate": [float, None]}

    def init_value(self, config, args):
        return True

    def mutate_value(self, value, config):
        if random() < getattr(config, self.mutate_rate_name):
            return random() < 0.5
        return value

class StringAttribute(BaseAttribute):
    _config_items = {"init_type": [str, None],
                     "mutate_rate": [float, 0.1]}

    _activations = ['ptanh', 'hardsigmoid', 'prelu', 'sigmoid']

    def init_value(self, config, args):
        init_type = getattr(config, self.init_type_name).lower()
        if init_type == 'random':
            return choice(self._activations)
        else:
            return init_type

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

        if random() < mutate_rate:
            return choice(self._activations)
        else:
            return value