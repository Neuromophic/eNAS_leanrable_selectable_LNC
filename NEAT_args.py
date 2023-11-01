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
EA_parser.add_argument('--compatibility_threshold',                 type=float,     default=8.0,            help='Compatibility threshold')

# Parameter configurations
EA_parser.add_argument('--gmin',                                    type=float,     default=0.1,            help='Type of bias initialization')
EA_parser.add_argument('--gmax',                                    type=float,     default=10,             help='Type of bias initialization')
EA_parser.add_argument('--theta_init_type',                         type=str,       default='g_init',       help='Type of weight initialization')
EA_parser.add_argument('--gb_init_type',                            type=str,       default='gb_init',      help='Type of bias initialization')
EA_parser.add_argument('--gd_init_type',                            type=str,       default='gd_init',      help='Type of gd initialization')

# Initialization of the nonlinear circuits
EA_parser.add_argument('--Activation_init_type',                    type=str,       default='random',       help='Type of initialization')
EA_parser.add_argument('--ACT_R1n_init_type',                       type=str,       default='ACT_R1n',      help='Type of initialization')
EA_parser.add_argument('--ACT_R2n_init_type',                       type=str,       default='ACT_R2n',      help='Type of initialization')
EA_parser.add_argument('--ACT_W1n_init_type',                       type=str,       default='ACT_W1n',      help='Type of initialization')
EA_parser.add_argument('--ACT_L1n_init_type',                       type=str,       default='ACT_L1n',      help='Type of initialization')
EA_parser.add_argument('--ACT_W2n_init_type',                       type=str,       default='ACT_W2n',      help='Type of initialization')
EA_parser.add_argument('--ACT_L2n_init_type',                       type=str,       default='ACT_L2n',      help='Type of initialization')
# learnable sigmoid circuits
EA_parser.add_argument('--S_R1n_init_type',                         type=str,       default='S_R1n',        help='Type of initialization')
EA_parser.add_argument('--S_R2n_init_type',                         type=str,       default='S_R2n',        help='Type of initialization')
EA_parser.add_argument('--S_W1n_init_type',                         type=str,       default='S_W1n',        help='Type of initialization')
EA_parser.add_argument('--S_L1n_init_type',                         type=str,       default='S_L1n',        help='Type of initialization')
EA_parser.add_argument('--S_W2n_init_type',                         type=str,       default='S_W2n',        help='Type of initialization')
EA_parser.add_argument('--S_L2n_init_type',                         type=str,       default='S_L2n',        help='Type of initialization')
# learnable pReLU circuits
EA_parser.add_argument('--ReLU_RHn_init_type',                      type=str,       default='ReLU_RHn',     help='Type of initialization') 
EA_parser.add_argument('--ReLU_RLn_init_type',                      type=str,       default='ReLU_RLn',     help='Type of initialization')
EA_parser.add_argument('--ReLU_RDn_init_type',                      type=str,       default='ReLU_RDn',     help='Type of initialization')
EA_parser.add_argument('--ReLU_RBn_init_type',                      type=str,       default='ReLU_RBn',     help='Type of initialization')
EA_parser.add_argument('--ReLU_Wn_init_type',                       type=str,       default='ReLU_Wn',      help='Type of initialization')
EA_parser.add_argument('--ReLU_Ln_init_type',                       type=str,       default='ReLU_Ln',      help='Type of initialization')
# learnable hard sigmoid circuits
EA_parser.add_argument('--HS_Rn_init_type',                         type=str,       default='HS_Rn',        help='Type of initialization')
EA_parser.add_argument('--HS_Wn_init_type',                         type=str,       default='HS_Wn',        help='Type of initialization')
EA_parser.add_argument('--HS_Ln_init_type',                         type=str,       default='HS_Ln',        help='Type of initialization')
# learnable negative weight circuits
EA_parser.add_argument('--NEG_R1n_init_type',                       type=str,       default='NEG_R1n',      help='Type of initialization')
EA_parser.add_argument('--NEG_R2n_init_type',                       type=str,       default='NEG_R2n',      help='Type of initialization')
EA_parser.add_argument('--NEG_R3n_init_type',                       type=str,       default='NEG_R3n',      help='Type of initialization')
EA_parser.add_argument('--NEG_W1n_init_type',                       type=str,       default='NEG_W1n',      help='Type of initialization')
EA_parser.add_argument('--NEG_L1n_init_type',                       type=str,       default='NEG_L1n',      help='Type of initialization')
EA_parser.add_argument('--NEG_W2n_init_type',                       type=str,       default='NEG_W2n',      help='Type of initialization')
EA_parser.add_argument('--NEG_L2n_init_type',                       type=str,       default='NEG_L2n',      help='Type of initialization')
EA_parser.add_argument('--NEG_W3n_init_type',                       type=str,       default='NEG_W3n',      help='Type of initialization')
EA_parser.add_argument('--NEG_L3n_init_type',                       type=str,       default='NEG_L3n',      help='Type of initialization')

# Connection add/remove rates
EA_parser.add_argument('--initial_connection',                      type=str,       default='unconnected',  help='initial connection')
EA_parser.add_argument('--conn_add_prob',                           type=float,     default=0.6,            help='Connection add probability')
EA_parser.add_argument('--conn_delete_prob',                        type=float,     default=0.4,            help='Connection delete probability')
EA_parser.add_argument('--enabled_mutate_rate',                     type=float,     default=0.1,            help='Rate of mutation to enable state')
# Node add/remove rates
EA_parser.add_argument('--node_add_prob',                           type=float,     default=0.6,            help='Node add probability')
EA_parser.add_argument('--node_delete_prob',                        type=float,     default=0.4,            help='Node delete probability')
