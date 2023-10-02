import argparse

EA_parser = argparse.ArgumentParser(description='NEAT Configurations')

# NEAT configurations
EA_parser.add_argument('--pop_size',                                type=int,       default=500,            help='Population size')
EA_parser.add_argument('--PATIENCE',                                type=int,       default=50,             help='Patience for early stopping')
EA_parser.add_argument('--GENERATION',                              type=int,       default=10**10,         help='Maximal generation for evolution')
EA_parser.add_argument('--species_elitism',                         type=int,       default=2,              help='minimal species in the whole population')
EA_parser.add_argument('--genome_elitism',                          type=int,       default=3,              help='Minimal genome in each species')
EA_parser.add_argument('--survival_threshold',                      type=float,     default=0.25,           help='Survival threshold')
EA_parser.add_argument('--max_stagnation',                          type=int,       default=20,             help='Patience for species stagnation (no improvement)')

# Species configurations
EA_parser.add_argument('--node_coefficient',                        type=float,     default=0.6,            help='Compatibility disjoint coefficient')
EA_parser.add_argument('--gene_coefficient',                        type=float,     default=0.4,            help='Compatibility weight coefficient')
EA_parser.add_argument('--compatibility_threshold',                 type=float,     default=2.0,            help='Compatibility threshold')

# Parameter configurations
EA_parser.add_argument('--gmin',                                    type=float,     default=0.1,            help='Type of bias initialization')
EA_parser.add_argument('--gmax',                                    type=float,     default=10,             help='Type of bias initialization')
EA_parser.add_argument('--theta_init_type',                         type=str,       default='g_init',       help='Type of weight initialization')
EA_parser.add_argument('--gb_init_type',                            type=str,       default='gb_init',      help='Type of bias initialization')
EA_parser.add_argument('--gd_init_type',                            type=str,       default='gd_init',      help='Type of gd initialization')
# Connection add/remove rates
EA_parser.add_argument('--initial_connection',                      type=str,       default='unconnected',  help='initial connection')
EA_parser.add_argument('--conn_add_prob',                           type=float,     default=0.6,            help='Connection add probability')
EA_parser.add_argument('--conn_delete_prob',                        type=float,     default=0.4,            help='Connection delete probability')
EA_parser.add_argument('--enabled_mutate_rate',                     type=float,     default=0.1,            help='Rate of mutation to enable state')
# Node add/remove rates
EA_parser.add_argument('--node_add_prob',                           type=float,     default=0.6,            help='Node add probability')
EA_parser.add_argument('--node_delete_prob',                        type=float,     default=0.4,            help='Node delete probability')


