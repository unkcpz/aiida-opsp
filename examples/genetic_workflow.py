from aiida.engine import run_get_node
from aiida import orm

from aiida_opsp.workflows.ga import GeneticAlgorithmWorkChain
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain
from aiida_opsp.workflows.ls import NelderMeadWorkChain

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
        'group': 's',
        'local_optimize': True,
    },
    's_qcut': {
        'key_name': 'angular_momentum_settings:s.qcut',
        'var_type': 'float',
        'space': {
            'range': [4.0, 10.0],
        },
        'group': 's',
        'local_optimize': True,
    },
    's_ncon': {
        'key_name': 'angular_momentum_settings:s.ncon',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
        },
        'group': 's',
        'local_optimize': False,
    },
    's_nbas': {
        'key_name': 'angular_momentum_settings:s.nbas',
        'var_type': 'int',
        'space': {
            'ref_to': 's_ncon',
            'range': [3, 5],
        },
        'group': 's',
        # 'local_optimize': False,
    },
    's_debl': {
        'key_name': 'angular_momentum_settings:s.debl',
        'var_type': 'float',
        'space': {
            'range': [0, 5.0],
        },
        'group': 's',
        'local_optimize': True,
    },
    'p_rc': {
        'key_name': 'angular_momentum_settings:p.rc',
        'var_type': 'float',
        'space': {
            'ref_to': 'rc(5)',
            'range': [0, 2.0],
        },
        'group': 'p',
        'local_optimize': True,
    },
    'p_qcut': {
        'key_name': 'angular_momentum_settings:p.qcut',
        'var_type': 'float',
        'space': {
            'range': [4.0, 10.0],
        },
        'group': 'p',
        'local_optimize': True,
    },
    'p_ncon': {
        'key_name': 'angular_momentum_settings:p.ncon',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
        },
        'group': 'p',
        'local_optimize': False,
    },
    'p_nbas': {
        'key_name': 'angular_momentum_settings:p.nbas',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
            'ref_to': 'p_ncon',
        },
        'group': 'p',
        # 'local_optimize': False,
    },
    'p_debl': {
        'key_name': 'angular_momentum_settings:p.debl',
        'var_type': 'float',
        'space': {
            'range': [0, 5.0],
        },
        'group': 'p',
        'local_optimize': True,
    },       
    'lpopt': {
        'key_name': 'local_potential_settings:lpopt',
        'var_type': 'int',
        'space': {
            'range': [1, 5],
        },
        'group': 'local',
        'local_optimize': False,
    },
    'rc(5)': {
        'key_name': 'local_potential_settings:rc(5)',
        'var_type': 'float',
        'space': {
            'range': [0.5, 3.0],
        },
        'group': 'local',
        'local_optimize': True,
    },       
    'dvloc0': {
        'key_name': 'local_potential_settings:dvloc0',
        'var_type': 'float',
        'space': {
            'range': [0.0, 3.0],
        },
        'group': 'local',
        'local_optimize': True,
    },    
}

def run():
    code = orm.load_code('oncv4@localhost')

    inputs = {
        'ga_parameters': orm.Dict(dict={
            'seed': 2023,
            'num_generations': 100,
            'num_individuals': 20,
            'num_mating_individuals': 8,
            'num_elite_individuals': 2,
            'num_new_individuals': 4,
            'num_offspring_individuals': 4,
            'max_thebest_count': 10,
            'elite_individual_mutate_probability': 0.4,
            'mediocre_individual_mutate_probability': 0.8,
        }),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'variable_info': orm.Dict(dict=variable_info),
        'result_key': orm.Str('result'),
        'fixture_inputs': {
            'code': code,
            'conf_name': conf_name,
            'lmax': orm.Int(1),
            'angular_momentum_settings': angular_momentum_settings,
            'local_potential_settings': local_potential_settings,
            'nlcc_settings': nlcc_settings,
            'run_atomic_test': orm.Bool(True),
            'dump_psp': orm.Bool(False),
        },
        'local_optimization_process': NelderMeadWorkChain,
        'local_optimization_parameters': orm.Dict(dict={
            'max_iter': 10,
            'xtol': 1e-3, # this is absolute tolerance the presicion for the input parameters
            'ftol': 1e-1, # this is relative tolerance for the score
        }),
    }
    res, node = run_get_node(GeneticAlgorithmWorkChain, **inputs)

    return res, node

if __name__ == '__main__':
    res, pk = run()
    # print(res['result'].get_dict(), pk)