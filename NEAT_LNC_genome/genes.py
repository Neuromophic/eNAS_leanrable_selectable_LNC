from random import random
from attributes import FloatAttribute, BoolAttribute, StringAttribute
import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(os.path.join(str(Path(os.getcwd()).parent), 'utils'))
import pLNC

class BaseGene(object):
    def __init__(self, key):
        self.key = key
        
    @classmethod
    def get_config_params(cls):
        params = []
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config, args):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config, args))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene

    def crossover(self, gene2, prob):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() < prob:
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

class PNCActivationGene(BaseGene):
    _gene_attributes = [StringAttribute('Activation'),
                        FloatAttribute('ACT_R1n'),
                        FloatAttribute('ACT_R2n'),
                        FloatAttribute('ACT_W1n'),
                        FloatAttribute('ACT_L1n'),
                        FloatAttribute('ACT_W2n'),
                        FloatAttribute('ACT_L2n'),
                        FloatAttribute('S_R1n'),
                        FloatAttribute('S_R2n'),
                        FloatAttribute('S_W1n'),
                        FloatAttribute('S_L1n'),
                        FloatAttribute('S_W2n'),
                        FloatAttribute('S_L2n'),
                        FloatAttribute('ReLU_RHn'),
                        FloatAttribute('ReLU_RLn'),
                        FloatAttribute('ReLU_RDn'),
                        FloatAttribute('ReLU_RBn'),
                        FloatAttribute('ReLU_Wn'),
                        FloatAttribute('ReLU_Ln'),
                        FloatAttribute('HS_Rn'),
                        FloatAttribute('HS_Wn'),
                        FloatAttribute('HS_Ln'),
                        FloatAttribute('NEG_R1n'),
                        FloatAttribute('NEG_R2n'),
                        FloatAttribute('NEG_R3n'),
                        FloatAttribute('NEG_W1n'),
                        FloatAttribute('NEG_L1n'),
                        FloatAttribute('NEG_W2n'),
                        FloatAttribute('NEG_L2n'),
                        FloatAttribute('NEG_W3n'),
                        FloatAttribute('NEG_L3n'),]

    def __init__(self):
        pass

    def init_attributes(self, config, args):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config, args))
        self.tanh = pLNC.TanhRT(args)
        self.HS = pLNC.HardSigmoidRT(args)
        self.ReLU = pLNC.pReLURT(args)
        self.sigmoid = pLNC.SigmoidRT(args)
        self.neg = pLNC.InvRT(args)

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))
        new_gene.tanh = self.tanh
        new_gene.HS = self.HS
        new_gene.ReLU = self.ReLU
        new_gene.sigmoid = self.sigmoid
        new_gene.neg = self.neg        
        return new_gene
        
    def distance(self, other, config):
        d_ptanh = abs(self.ACT_R1n - other.ACT_R1n) + abs(self.ACT_R2n - other.ACT_R2n) +\
                  abs(self.ACT_W1n - other.ACT_W1n) + abs(self.ACT_L1n - other.ACT_L1n) +\
                  abs(self.ACT_W2n - other.ACT_W2n) + abs(self.ACT_L2n - other.ACT_L2n)
        d_sigmoid = abs(self.S_R1n - other.S_R1n) + abs(self.S_R2n - other.S_R2n) +\
                    abs(self.S_W1n - other.S_W1n) + abs(self.S_L1n - other.S_L1n) +\
                    abs(self.S_W2n - other.S_W2n) + abs(self.S_L2n - other.S_L2n)
        d_prelu = abs(self.ReLU_RHn - other.ReLU_RHn) + abs(self.ReLU_RLn - other.ReLU_RLn) +\
                  abs(self.ReLU_RDn - other.ReLU_RDn) + abs(self.ReLU_RBn - other.ReLU_RBn) +\
                  abs(self.ReLU_Wn - other.ReLU_Wn) + abs(self.ReLU_Ln - other.ReLU_Ln)
        d_hs = abs(self.HS_Rn - other.HS_Rn) + abs(self.HS_Wn - other.HS_Wn) + abs(self.HS_Ln - other.HS_Ln)
        d_neg = abs(self.NEG_R1n - other.NEG_R1n) + abs(self.NEG_R2n - other.NEG_R2n) + abs(self.NEG_R3n - other.NEG_R3n) +\
                abs(self.NEG_W1n - other.NEG_W1n) + abs(self.NEG_L1n - other.NEG_L1n) +\
                abs(self.NEG_W2n - other.NEG_W2n) + abs(self.NEG_L2n - other.NEG_L2n) +\
                abs(self.NEG_W3n - other.NEG_W3n) + abs(self.NEG_L3n - other.NEG_L3n)
        
        w_ptanh, w_sigmoid, w_prelu, w_hs = 1., 1., 1., 1.
        if self.Activation == 'ptanh':
            w_ptanh += 1.
        elif self.Activation == 'sigmoid':
            w_sigmoid += 1.
        elif self.Activation == 'prelu':
            w_prelu += 1.
        elif self.Activation == 'hardsigmoid':
            w_hs += 1.
        
        if other.Activation == 'ptanh':
            w_ptanh += 1.
        elif other.Activation == 'sigmoid':
            w_sigmoid += 1.
        elif other.Activation == 'prelu':
            w_prelu += 1.
        elif other.Activation == 'hardsigmoid':
            w_hs += 1.

        weight_ptanh = w_ptanh / (w_ptanh + w_sigmoid + w_prelu + w_hs)
        weight_sigmoid = w_sigmoid / (w_ptanh + w_sigmoid + w_prelu + w_hs)
        weight_prelu = w_prelu / (w_ptanh + w_sigmoid + w_prelu + w_hs)
        weight_hs = w_hs / (w_ptanh + w_sigmoid + w_prelu + w_hs)

        d_value = weight_ptanh * d_ptanh + weight_sigmoid * d_sigmoid + weight_prelu * d_prelu + weight_hs * d_hs + d_neg
        
        d_act = 0. if self.Activation == other.Activation else 1.
        return d_value * config.gene_coefficient + d_act
    
    def crossover(self, gene2, prob):
        new_gene = self.__class__()
        for a in self._gene_attributes:
            if random() < prob:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))
        new_gene.tanh = self.tanh
        new_gene.HS = self.HS
        new_gene.ReLU = self.ReLU
        new_gene.sigmoid = self.sigmoid
        new_gene.neg = self.neg      
        return new_gene
