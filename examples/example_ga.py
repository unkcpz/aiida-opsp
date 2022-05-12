from aiida.engine import run_get_pk
from aiida.engine import calcfunction
from aiida import orm

from aiida_opsp.workflows.ga import GeneticAlgorithmWorkChain
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

def run():
    code = orm.load_code('oncv4@localhost0')

    conf_name = orm.Str('Li-s')
    angular_momentum_settings = orm.Dict(
        dict={
            's': {
                'ncon': 4,
                'nbas': 8,
                'nproj': 2,
                'debl': 1.0,
            },
            'p': {
                'rc': 1.1,
                'ncon': 4,
                'nbas': 8,
                'qcut': 9.0,
                'nproj': 2,
                'debl': 1.0,
            }, 
        }
    )
    local_potential_settings = orm.Dict(
        dict={
            'llcol': 4, # fix
            'lpopt': 5, # 1-5, algorithm enum set
            'rc(5)': 1.1,
            'dvloc0': 0.0,
        }
    )
    nlcc_settings = orm.Dict(
        dict={
            'icmod': 0,
            'fcfact': 0.25,
        }
    )
    
    inputs = {
        'parameters': orm.Dict(dict={
            'num_generation': 20,
            'num_pop_per_generation': 20,
            'num_genes': 2, # check shape compatible with gene_space
            'num_mating_parents': 16,
            'num_elitism': 2,
            'num_mutation_genes': 2,    # not being used
            'individual_mutate_probability': 1.0,
            'gene_mutate_probability': 0.2,
            'gene_space': [{'low': 1.0, 'high': 2.0}, {'low': 4.0, 'high': 14.0}],
            'gene_type': ['float', 'float'],
            'seed': 979,
        }),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'input_nested_keys': orm.List(list=[
            'angular_momentum_settings:s.rc',
            'angular_momentum_settings:s.qcut',
            # 'angular_momentum_settings:p.qcut',
        ]
        ),
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