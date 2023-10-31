"""Handles genomes (individuals in the population)."""
import copy
from itertools import count
from random import choice, random, shuffle
from config import ConfigParameter
from genes import PNCNodeGene, PNCConnectionGene, PNCActivationGene
from feed_forward import creates_cycle
from feed_forward import required_for_output


class GenomeConfig(object):
    def __init__(self, params):
        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('node_coefficient', float),
                        ConfigParameter('gene_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('gmin', float),
                        ConfigParameter('gmax', float)]

        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.activation_gene_type = params['activation_gene_type']
        self._params += self.activation_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.node_indexer = None

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(node_dict)) + 1)
        new_id = next(self.node_indexer)
        return new_id


class PNCGenome(object):
    ''' the genome model of a printed neuromorphic circuit '''
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = PNCNodeGene
        param_dict['activation_gene_type'] = PNCActivationGene
        param_dict['connection_gene_type'] = PNCConnectionGene
        return GenomeConfig(param_dict)

    def __init__(self, key):
        self.key = key
        self.connections = {}
        self.nodes = {}
        self.fitness = None

    def configure_new(self, config, args):
        self.activation_node = config.activation_gene_type()
        self.activation_node.init_attributes(config, args)
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, args, node_key)

    def crossover(self, genome1, genome2):
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        cross_prob = parent1.fitness / (parent1.fitness + parent2.fitness)

        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                self.connections[key] = cg1.copy()
            else:
                
                self.connections[key] = cg1.crossover(cg2, cross_prob)

        parent1_set = parent1.nodes
        parent2_set = parent2.nodes
        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            if ng2 is None:
                self.nodes[key] = ng1.copy()
            else:
                self.nodes[key] = ng1.crossover(ng2, cross_prob)
        
        self.activation_node = parent1.activation_node.crossover(parent2.activation_node, cross_prob)

    def mutate(self, config, args):
        if random() < config.node_add_prob:
            self.mutate_add_node(config, args)
        if random() < config.node_delete_prob:
            self.mutate_delete_node(config)
        if random() < config.conn_add_prob:
            self.mutate_add_connection(config, args)
        if random() < config.conn_delete_prob:
            self.mutate_delete_connection()

        for cg in self.connections.values():
            cg.mutate(config)
        for ng in self.nodes.values():
            ng.mutate(config)
        self.activation_node.mutate(config)

    def mutate_add_node(self, config, args):
        if not self.connections:
            return

        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, args, new_node_id)
        self.nodes[new_node_id] = ng

        conn_to_split.enabled = False
        i, o = conn_to_split.key
        self.add_connection(config, args, i, new_node_id, conn_to_split.theta)
        self.add_connection(config, args, new_node_id, o, config.gmax)

    def add_connection(self, config, args, input_key, output_key, theta=None):
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config, args)
        if theta is not None:
            connection.theta = theta
        self.connections[key] = connection

    def mutate_add_connection(self, config, args):
        possible_outputs = list(self.nodes)
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        key = (in_node, out_node)
        if key in self.connections:
            return

        if creates_cycle(list(self.connections), key):
            return

        cg = self.create_connection(config, args, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        available_nodes = [
            k for k in self.nodes if k not in config.output_keys]
        if not available_nodes:
            return

        del_key = choice(available_nodes)
        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)
        for key in connections_to_delete:
            del self.connections[key]
        del self.nodes[del_key]

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1
            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + (config.node_coefficient * disjoint_nodes)) / max_nodes

        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance + (config.node_coefficient * disjoint_connections)) / max_conn

        activation_distance = self.activation_node.distance(other.activation_node, config)

        distance = node_distance + connection_distance + activation_distance
        return distance

    def size(self):
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    @staticmethod
    def create_node(config, args, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config, args)
        return node

    @staticmethod
    def create_connection(config, args, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config, args)
        return connection

    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, self.connections,
                                                                  genome_config.input_keys,
                                                                  genome_config.output_keys,
                                                                  genome_config.gmin)
        new_genome = PNCGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        new_genome.activation_node = copy.deepcopy(self.activation_node)
        return new_genome


def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys, gmin):
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            if abs(cg.theta) > gmin:
                used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes
