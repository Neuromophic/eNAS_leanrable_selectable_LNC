import numpy


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
    def __init__(self, inputs, outputs, node_evals, args):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.args = args

    def act(self, a):
        return self.args.etaA1 + self.args.etaA2 * numpy.tanh((a - self.args.etaA3) * self.args.etaA4)

    def neg(self, x):
        return -(self.args.etaN1 + self.args.etaN2 * numpy.tanh((x - self.args.etaN3) * self.args.etaN4))

    @staticmethod
    def theta_ste(m, config):
        theta_temp_ = numpy.array(numpy.clip(m, -config.genome_config.gmax, config.genome_config.gmax))
        theta_temp_[numpy.abs(theta_temp_) < config.genome_config.gmin] = 0
        return theta_temp_

    def forward(self, data, config):
        self.values = {key: numpy.zeros(
            data.shape[0]) for key in self.input_nodes + self.output_nodes}
        data = numpy.array(data).T.tolist()
        for k, v in zip(self.input_nodes, data):
            self.values[k] = numpy.array(v)

        for node, gb, gd, inputs in self.node_evals:
            theta_ = numpy.array([input[1] for input in inputs] + [gb, gd])
            theta = self.theta_ste(theta_, config)

            node_inputs = []
            for i, g in inputs:
                value_temp = self.neg(self.values[i])*(self.theta_ste(g, config) < 0) + self.values[i]*(self.theta_ste(g, config) >= 0)
                node_inputs.append(value_temp * (abs(self.theta_ste(g, config)) / numpy.sum(abs(theta))))
            bias_temp = self.neg(1.)*(self.theta_ste(gb, config) < 0) + 1.*(self.theta_ste(gb, config) >= 0)
            mac = sum(node_inputs) + bias_temp * (abs(self.theta_ste(gb, config)) / numpy.sum(abs(theta)))

            self.values[node] = self.act(mac)

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

        return PrintedNeuromorphicCircuit(config.genome_config.input_keys, config.genome_config.output_keys, node_evals, args)
