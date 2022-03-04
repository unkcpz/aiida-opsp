from aiida import engine
from aiida import orm

from aiida_opsp.calcjob import OncvPseudoCalculation

computer = orm.load_computer('localhost')
code = orm.Code(label='oncv4-0', remote_computer_exec=[computer, '/home/jyu/Projects/WP-OPSP/bin/oncvpsp.x'], input_plugin_name='opsp.pseudo.oncv')

input_parameters = {
    'atom_info': ['Li', 3, 0, 2, 3, 'upf'],
    'atom_conf': [
            [1, 0, 2.0],
            [2, 0, 1.0],
        ],
    'opt_lmax': 1,
    'opt_rrkj': [
            [0, 2.0, 0,0, 4, 8, 6.0],
            [1, 3.0, 0.1, 4, 8, 4.0],
        ],
    'local_potential': [4, 5, 1.0, 0.0],
    'projectors': [
        [0, 2, 0.0],
        [1, 2, 0.75],    
    ],
    'core_corr': [0, 0.0],
}

inputs = {
    'code': code,
    'input': orm.Dict(dict=input_parameters),
    'metadata': {
        'options': {
            'withmpi': False,
            'parser_name': 'opsp.pseudo.oncv',
        },
        # 'dry_run': True,
    }
}

output, node = engine.run_get_node(OncvPseudoCalculation, **inputs)
print(output, node)