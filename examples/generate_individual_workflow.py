from aiida import orm
from aiida.engine import run_get_node

from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain
from aiida_opsp.workflows.individual import GenerateValidIndividual, GenerateMutateValidIndividual

conf_name = orm.Str('Li-low')
angular_momentum_settings = orm.Dict(
    dict={
        's': {
            'nproj': 2,
        },
        'p': {
            'nproj': 2,
        }, 
    }
)
local_potential_settings = orm.Dict(
    dict={
        'llcol': 4, # fix
    }
)
nlcc_settings = orm.Dict(
    dict={
        'icmod': 3,
        'fcfact': 5.0,
        'rcfact': 1.4,
    }
)

variable_info = {
    's_rc': {
        'key_name': 'angular_momentum_settings:s.rc',
        'var_type': 'float',
        'space': {
            'ref_to': 'rc(5)',
            'range': [0, 2.0],
        },
        'local_optimize': True,
    },
    's_qcut': {
        'key_name': 'angular_momentum_settings:s.qcut',
        'var_type': 'float',
        'space': {
            'range': [4.0, 10.0],
        },
        'local_optimize': True,
    },
    's_ncon': {
        'key_name': 'angular_momentum_settings:s.ncon',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
        },
        'local_optimize': False,
    },
    's_nbas': {
        'key_name': 'angular_momentum_settings:s.nbas',
        'var_type': 'int',
        'space': {
            'ref_to': 's_ncon',
            'range': [3, 5],
        },
        # 'local_optimize': False,
    },
    's_debl': {
        'key_name': 'angular_momentum_settings:s.debl',
        'var_type': 'float',
        'space': {
            'range': [0, 5.0],
        },
        'local_optimize': True,
    },
    'p_rc': {
        'key_name': 'angular_momentum_settings:p.rc',
        'var_type': 'float',
        'space': {
            'ref_to': 'rc(5)',
            'range': [0, 2.0],
        },
        'local_optimize': True,
    },
    'p_qcut': {
        'key_name': 'angular_momentum_settings:p.qcut',
        'var_type': 'float',
        'space': {
            'range': [4.0, 10.0],
        },
        'local_optimize': True,
    },
    'p_ncon': {
        'key_name': 'angular_momentum_settings:p.ncon',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
        },
        'local_optimize': False,
    },
    'p_nbas': {
        'key_name': 'angular_momentum_settings:p.nbas',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
            'ref_to': 'p_ncon',
        },
        # 'local_optimize': False,
    },
    'p_debl': {
        'key_name': 'angular_momentum_settings:p.debl',
        'var_type': 'float',
        'space': {
            'range': [0, 5.0],
        },
        'local_optimize': True,
    },       
    'lpopt': {
        'key_name': 'local_potential_settings:lpopt',
        'var_type': 'int',
        'space': {
            'range': [1, 5],
        },
        'local_optimize': False,
    },
    'rc(5)': {
        'key_name': 'local_potential_settings:rc(5)',
        'var_type': 'float',
        'space': {
            'range': [0.5, 3.0],
        },
        'local_optimize': True,
    },       
    'dvloc0': {
        'key_name': 'local_potential_settings:dvloc0',
        'var_type': 'float',
        'space': {
            'range': [0.0, 3.0],
        },
        'local_optimize': True,
    },    
}

def run():
    code = orm.load_code('oncv4@localhost')

    inputs = {
        'evaluate_process': OncvPseudoBaseWorkChain,
        'variable_info': orm.Dict(variable_info),
        'fixture_inputs': {
            'code': code,
            'conf_name': conf_name,
            'lmax': orm.Int(1),
            'angular_momentum_settings': angular_momentum_settings,
            'local_potential_settings': local_potential_settings,
            'nlcc_settings': nlcc_settings,
            'run_atomic_test': orm.Bool(False),
            'dump_psp': orm.Bool(False),
        }
    }

    _, node = run_get_node(GenerateValidIndividual, **inputs)
    
    individual = node.outputs.final_individual
    print(individual.get_dict())

    # Run mutate the individual
    inputs['probability'] = orm.Float(0.5)
    inputs['init_individual'] = individual
    
    _, node = run_get_node(GenerateMutateValidIndividual, **inputs)

    print(node.outputs.final_individual.get_dict())

    
if __name__ == '__main__':
    run()