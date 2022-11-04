from aiida.engine import run_get_pk
from aiida.engine import calcfunction
from aiida import orm

from aiida_opsp.workflows.ga import GeneticAlgorithmWorkChain
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

def run():
    code = orm.load_code('oncv4@localhost1')

    conf_name = orm.Str('Li-low')
    angular_momentum_settings = orm.Dict(
        dict={
            's': {
                'rc': 2.972,
                'qcut': 4.5706851858103,
                'ncon': 4,
                'nbas': 8,
                'nproj': 2,
                'debl': 0.2354,
            },
            'p': {
                'rc': 3.0142209418377,
                'qcut': 7.9012663432424,
                'ncon': 4,
                'nbas': 7,
                'nproj': 2,
                'debl': 1.782067837456,
            }, 
        }
    )
    local_potential_settings = orm.Dict(
        dict={
            'llcol': 4, # fix
            'lpopt': 3, # 1-5, algorithm enum set
            'rc(5)': 1.7539158159146,
            'dvloc0': 1.4779768452039,
        }
    )
    nlcc_settings = orm.Dict(
        dict={
            # 'icmod': 3,
            # 'fcfact': 5.0,
            # 'rcfact': 1.4,
        }
    )
    
    inputs = {
        'parameters': orm.Dict(dict={
            'num_generation': 10,
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
            'icmod': {
                'key_name': 'nlcc_settings:icmod',
                'type': 'int',
                'space': {
                    'low': 3,
                    'high': 4,
                },
                'local_optimize': False,
            },
            'fcfact': {
                'key_name': 'nlcc_settings:fcfact',
                'type': 'float',
                'space': {
                    'low': 0.0, 
                    'high': 10.0,
                },
                'local_optimize': True,
            },
            'rcfact': {
                'key_name': 'nlcc_settings:rcfact',
                'type': 'float',
                'space': {
                    'low': 0.0, 
                    'high': 5.0,
                },
                'local_optimize': True,
            },
        }),
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
        }
    }
    res, pk = run_get_pk(GeneticAlgorithmWorkChain, **inputs)

    return res, pk

if __name__ == '__main__':
    res, pk = run()
    # print(res['result'].get_dict(), pk)