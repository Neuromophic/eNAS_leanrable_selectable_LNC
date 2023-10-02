from random import gauss, random, uniform
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

    def init_value(self, config):
        init_type = getattr(config, self.init_type_name).lower()
        if 'g_init' in init_type:
            return uniform(0.1, 1.)
        if 'gb_init' in init_type:
            return uniform(0.1, 1.)
        if 'gd_init' in init_type:
            return config.gmax - uniform(0, 1)

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)
        mutate_power = getattr(config, self.mutate_power_name)

        if random() < mutate_rate:
            return value + gauss(0., mutate_power)
        else:
            return value


class BoolAttribute(BaseAttribute):
    _config_items = {"mutate_rate": [float, None]}

    def init_value(self, config):
        return True

    def mutate_value(self, value, config):
        if random() < getattr(config, self.mutate_rate_name):
            return random() < 0.5
        return value
