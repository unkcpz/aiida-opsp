from aiida.engine import run_get_pk
from aiida.engine import calcfunction
from aiida import orm

from aiida_opsp.workflows.ga import GeneticAlgorithmWorkChain
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

def run():
    code = orm.load_code('oncv4@localhost')

    conf_name = orm.Str('Ga')
    angular_momentum_settings = orm.Dict(
        dict={
            's': {
                # 'rc': 1.4,
                # 'qcut': 8.6,
                # 'ncon': 3,
                # 'nbas': 7,
                'nproj': 2,
                # 'debl': 2.0,
            },
            'p': {
                # 'rc': 2.2,
                # 'qcut': 7.0,
                # 'ncon': 3,
                # 'nbas': 7,
                'nproj': 2,
                # 'debl': 3.5,
            }, 
            'd': {
                # 'rc': 2.2,
                # 'qcut': 7.0,
                # 'ncon': 3,
                # 'nbas': 7,
                'nproj': 2,
                # 'debl': 3.5,
            }, 
        }
    )
    local_potential_settings = orm.Dict(
        dict={
            'llcol': 4, # fix
            # 'lpopt': 5, # 1-5, algorithm enum set
            # 'rc(5)': 2.2,
            # 'dvloc0': 2.0,
        }
    )
    nlcc_settings = orm.Dict(
        dict={
            'icmod': 3,
            'fcfact': 5.0,
            'rcfact': 1.4,
        }
    )
    
    inputs = {
        'parameters': orm.Dict(dict={
            'num_generation': 20,
            'num_pop_per_generation': 20,
            'num_mating_parents': 16,
            'num_elitism': 2,
            'individual_mutate_probability': 1.0,
            'gene_mutate_elitism_probability': 0.9, # high mutate rate since using gaussian for mutate
            'gene_mutate_mediocrity_probability': 0.4,
            'seed': 979,
            'local_search_base_parameters': {
                'max_iter': 10,
                'xtol': 1e-1,
                'ftol': 1e-1,
            }
        }),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'vars_info': orm.Dict(dict={
            's_rc': {
                'key_name': 'angular_momentum_settings:s.rc',
                'type': 'float',
                'space': {
                    'refto': 'rc(5)',
                    'low': 0, 
                    'high': 2.0,
                },
                'local_optimize': True,
            },
            's_qcut': {
                'key_name': 'angular_momentum_settings:s.qcut',
                'type': 'float',
                'space': {
                    'low': 4.0, 
                    'high': 10.0,
                },
                'local_optimize': True,
            },
            's_ncon': {
                'key_name': 'angular_momentum_settings:s.ncon',
                'type': 'int',
                'space': {
                    'low': 3, 
                    'high': 5,
                },
                'local_optimize': False,
            },
            's_nbas': {
                'key_name': 'angular_momentum_settings:s.nbas',
                'type': 'int',
                'space': {
                    'refto': 's_ncon',
                    'low': 3, 
                    'high': 5,
                },
                # 'local_optimize': False,
            },
            # 'angular_momentum_settings:s.nproj',
            's_debl': {
                'key_name': 'angular_momentum_settings:s.debl',
                'type': 'float',
                'space': {
                    'low': 0, 
                    'high': 5.0,
                },
                'local_optimize': True,
            },
            'p_rc': {
                'key_name': 'angular_momentum_settings:p.rc',
                'type': 'float',
                'space': {
                    'refto': 'rc(5)',
                    'low': 0, 
                    'high': 2.5,
                },
                'local_optimize': True,
            },
            'p_qcut': {
                'key_name': 'angular_momentum_settings:p.qcut',
                'type': 'float',
                'space': {
                    'low': 4.0, 
                    'high': 10.0,
                },
                'local_optimize': True,
            },
            'p_ncon': {
                'key_name': 'angular_momentum_settings:p.ncon',
                'type': 'int',
                'space': {
                    'low': 3, 
                    'high': 5,
                },
                'local_optimize': False,
            },
            'p_nbas': {
                'key_name': 'angular_momentum_settings:p.nbas',
                'type': 'int',
                'space': {
                    'refto': 'p_ncon',
                    'low': 3, 
                    'high': 5,
                },
                # 'local_optimize': False,
            },
            'p_debl': {
                'key_name': 'angular_momentum_settings:p.debl',
                'type': 'float',
                'space': {
                    'low': 0, 
                    'high': 5.0,
                },
                'local_optimize': True,
            },
            'd_rc': {
                'key_name': 'angular_momentum_settings:d.rc',
                'type': 'float',
                'space': {
                    'refto': 'rc(5)',
                    'low': 0, 
                    'high': 2.5,
                },
                'local_optimize': True,
            }, 
            'd_qcut': {
                'key_name': 'angular_momentum_settings:d.qcut',
                'type': 'float',
                'space': {
                    'low': 4.0, 
                    'high': 10.0,
                },
                'local_optimize': True,
            },
            'd_ncon': {
                'key_name': 'angular_momentum_settings:d.ncon',
                'type': 'int',
                'space': {
                    'low': 3, 
                    'high': 5,
                },
                'local_optimize': False,
            },
            'd_nbas': {
                'key_name': 'angular_momentum_settings:d.nbas',
                'type': 'int',
                'space': {
                    'refto': 'd_ncon',
                    'low': 3, 
                    'high': 5,
                },
                # 'local_optimize': False,
            },
            'd_debl': {
                'key_name': 'angular_momentum_settings:d.debl',
                'type': 'float',
                'space': {
                    'low': 0, 
                    'high': 5.0,
                },
                'local_optimize': True,
            },       
            'lpopt': {
                'key_name': 'local_potential_settings:lpopt',
                'type': 'int',
                'space': {
                    'low': 1, 
                    'high': 5,
                },
                'local_optimize': False,
            },
            'rc(5)': {
                'key_name': 'local_potential_settings:rc(5)',
                'type': 'float',
                'space': {
                    'low': 0.5, 
                    'high': 3.0,
                },
                'local_optimize': True,
            },       
            'dvloc0': {
                'key_name': 'local_potential_settings:dvloc0',
                'type': 'float',
                'space': {
                    'low': 0, 
                    'high': 3.0,
                },
                'local_optimize': True,
            },                
        }),
        'result_key': orm.Str('result'),
        'fixture_inputs': {
            'code': code,
            'conf_name': conf_name,
            'lmax': orm.Int(2),
            'angular_momentum_settings': angular_momentum_settings,
            'local_potential_settings': local_potential_settings,
            'nlcc_settings': nlcc_settings,
            'run_atomic_test': orm.Bool(True),
            'dump_psp': orm.Bool(False),
        }
    }
    res, pk = run_get_pk(GeneticAlgorithmWorkChain, **inputs)

    return res, pk

if __name__ == '__main__':
    res, pk = run()
    # print(res['result'].get_dict(), pk)