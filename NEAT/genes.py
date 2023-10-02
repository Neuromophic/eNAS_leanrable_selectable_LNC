from random import random
from attributes import FloatAttribute, BoolAttribute


class BaseGene(object):
    def __init__(self, key):
        self.key = key
        
    @classmethod
    def get_config_params(cls):
        params = []
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene

    def crossover(self, gene2):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.4:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))
        return new_gene


class PNCNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('gb'),
                        FloatAttribute('gd')]

    def __init__(self, key):
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.gb - other.gb) + abs(self.gd - other.gd)
        return d * config.gene_coefficient


class PNCConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('theta'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.theta - other.theta)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.gene_coefficient
