import numpy
import torch
import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(os.path.join(str(Path(os.getcwd()).parent), 'utils'))
import pLNC

def creates_cycle(connections, test):
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True
                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    required = set(outputs)
    checked = set(outputs)
    while True:
        checking = set(a for (a, b) in connections if b in checked and a not in checked)

        if not checking:
            break
        layer_nodes = set(x for x in checking if x not in inputs)

        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        checked = checked.union(checking)

    return required


def feed_forward_layers(inputs, outputs, connections):
    required = required_for_output(inputs, outputs, connections)

    layers = []
    checked = set(inputs)
    while True:
        candidate = set(
            b for (a, b) in connections if a in checked and b not in checked)

        collection = set()
        for key in candidate:
            if key in required and all(a in checked for (a, b) in connections if b == key):
                collection.add(key)

        if not collection:
            break

        layers.append(collection)
        checked = checked.union(collection)

    return layers


class PrintedNeuromorphicCircuit:
    def __init__(self, inputs, outputs, node_evals, activation_node, args):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.activation_node = activation_node
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train

    def act(self, a):
        activation_fn = self.activation_node.Activation
        if activation_fn == 'ptanh':
            act = pLNC.TanhRT(self.args)
            act.N = self.N
            act.epsilon = self.epsilon
            act.rt_.data = torch.tensor([self.activation_node.ACT_R1n, self.activation_node.ACT_R2n,
                                         self.activation_node.ACT_W1n, self.activation_node.ACT_L1n,
                                         self.activation_node.ACT_W2n, self.activation_node.ACT_L2n])
        elif activation_fn == 'hardsigmoid':
            act = pLNC.HardSigmoidRT(self.args)
            act.N = self.N
            act.epsilon = self.epsilon
            act.rt_.data = torch.tensor([self.activation_node.HS_Rn, self.activation_node.HS_Wn, self.activation_node.HS_Ln])
        elif activation_fn == 'prelu':
            act = pLNC.pReLURT(self.args)
            act.N = self.N
            act.epsilon = self.epsilon
            act.rt_.data = torch.tensor([self.activation_node.ReLU_RHn, self.activation_node.ReLU_RLn,
                                         self.activation_node.ReLU_RDn, self.activation_node.ReLU_RBn,
                                         self.activation_node.ReLU_Wn, self.activation_node.ReLU_Ln])
        elif activation_fn == 'sigmoid':
            act = pLNC.SigmoidRT(self.args)
            act.N = self.N
            act.epsilon = self.epsilon
            act.rt_.data = torch.tensor([self.activation_node.S_R1n, self.activation_node.S_R2n,
                                         self.activation_node.S_W1n, self.activation_node.S_L1n,
                                         self.activation_node.S_W2n, self.activation_node.S_L2n])
        else:
            raise ValueError(f'Unknown activation: {self.activation}')
        
        return act(a)
    
    def neg(self, x):
        inv = pLNC.InvRT(self.args)
        inv.N = self.N
        inv.epsilon = self.epsilon
        inv.rt_.data = torch.tensor([self.activation_node.NEG_R1n, self.activation_node.NEG_R2n, self.activation_node.NEG_R3n,
                                     self.activation_node.NEG_W1n, self.activation_node.NEG_L1n,
                                     self.activation_node.NEG_W2n, self.activation_node.NEG_L2n,
                                     self.activation_node.NEG_W3n, self.activation_node.NEG_L3n])
        return inv(x)

    @staticmethod
    def theta_ste(m, config):
        theta_temp_ = numpy.array(numpy.clip(m, -config.genome_config.gmax, config.genome_config.gmax))
        theta_temp_[numpy.abs(theta_temp_) < config.genome_config.gmin] = 0
        return theta_temp_
    
    def theta_ste_noisy(self, m, config):
        theta = self.theta_ste(m, config)
        theta = numpy.vstack([theta] * self.N)
        noise = (numpy.random.rand(theta.shape[0], theta.shape[1]) * 2. - 1.) * self.epsilon + 1.
        return  theta * noise

    def forward(self, data, config):
        
        # data = data.repeat(self.N, 1, 1)
        # print('data_shape', data.shape)
        self.values = {key: numpy.zeros([self.N, data.shape[0]]) for key in self.input_nodes + self.output_nodes}
        # print('value_shape', self.values[self.input_nodes[0]].shape)
        data = data.T.tolist()

        for k, v in zip(self.input_nodes, data):
            # print('single_data', len(v))
            result = numpy.repeat(numpy.array(v)[numpy.newaxis, :], self.N, axis=0)
            self.values[k] = result
            # print('single_value', result.shape)

        for node, gb, gd, inputs in self.node_evals:
            theta_ = numpy.array([input[1] for input in inputs] + [gb, gd])
            theta = self.theta_ste_noisy(theta_, config)
            # print('theta_ shape', theta_.shape)
            # print('theta shape', theta.shape)

            node_inputs = []
            for i, g in inputs:
                # print('g', self.theta_ste(g, config).shape)
                # print('g_noisy_shape', self.theta_ste_noisy(g, config).shape)
                # print('value', self.values[i].shape)
                # print('value_shape', torch.tensor(self.values[i]).unsqueeze(-1).shape)
                # print('negative_g_shape', (self.theta_ste(g, config) < 0).shape)
                # print('negative_g_noisy_shape', (self.theta_ste_noisy(g, config) < 0).shape)
                # print('after neg', self.neg(torch.tensor(self.values[i]).unsqueeze(-1))[:,:,0].shape)
                

                value_temp = self.neg(torch.tensor(self.values[i]).unsqueeze(-1))[:,:,0].detach().numpy() * (self.theta_ste(g, config) < 0) +\
                             self.values[i]*(self.theta_ste(g, config) >= 0)
                # print('value_temp', value_temp.shape)
                # print(abs(self.theta_ste_noisy(g, config)).shape)
                # print(abs(theta).shape)
                weight = abs(self.theta_ste_noisy(g, config)) / (numpy.sum(abs(theta)).reshape(-1,1) + 1e-8)
                # print('weight', weight.shape)
                # print((value_temp * weight).shape)
                node_inputs.append(value_temp * weight)
            
            # print('neg_one', (self.neg(torch.ones(self.N, 1, 1))[:,:,0].detach().numpy()).shape)
            # print('neg_mask', (self.theta_ste_noisy(gb, config) < 0).shape)
            bias_temp = self.neg(torch.ones(self.N, 1, 1))[:,:,0].detach().numpy() * (self.theta_ste_noisy(gb, config) < 0) + 1. *(self.theta_ste_noisy(gb, config) >= 0)
            # print('bias_temp', bias_temp.shape)
            mac = numpy.array(sum(node_inputs)) + bias_temp * (abs(self.theta_ste_noisy(gb, config))) / (numpy.sum(abs(theta)).reshape(-1,1) + 1e-8)
            # print('mac', mac.shape)

            z = torch.tensor(mac).unsqueeze(-1).float()
            self.values[node] = self.act(z)[:,:,0].detach().numpy()
            # print(self.values[node].shape)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config, args):
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)

        node_evals = [
            (
                node,
                genome.nodes[node].gb,
                genome.nodes[node].gd,
                [(conn_key[0], genome.connections[conn_key].theta)
                 for conn_key in connections if conn_key[1] == node]
            )
            for layer in layers for node in layer
        ]

        return PrintedNeuromorphicCircuit(config.genome_config.input_keys, config.genome_config.output_keys, node_evals, genome.activation_node, args)
