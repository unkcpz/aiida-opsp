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
                # 'rc': 1.4,
                # 'qcut': 8.6,
                # 'ncon': 3,
                # 'nbas': 7,
                'nproj': 2,
                # 'debl': 2.0,
            },
            'p': {
                'rc': 2.2,
                'qcut': 7.0,
                'ncon': 3,
                'nbas': 7,
                'nproj': 2,
                'debl': 3.5,
            }, 
        }
    )
    local_potential_settings = orm.Dict(
        dict={
            'llcol': 4, # fix
            'lpopt': 5, # 1-5, algorithm enum set
            'rc(5)': 2.2,
            'dvloc0': 2.0,
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
            'num_generation': 2,
            'num_pop_per_generation': 10,
            'num_genes': 4, # check shape compatible with gene_space
            'num_mating_parents': 6,
            'num_elitism': 2,
            'individual_mutate_probability': 1.0,
            'gene_mutate_probability': 0.4,
            'gene_space': [
                {'low': 2.2, 'high': 3.0}, # s, rc
                {'low': 4.0, 'high': 10.0}, # s, qcut
                {'low': 3, 'high': 5}, # s, ncon
                {'low': 5, 'high': 10}, # s, nbas
                # {'low': 1, 'high': 2},  # s, nproj 
                {'low': 0.0, 'high': 3.0}, # s, debl
            ],
            'gene_type': [
                'float', 
                'float',
                'int', 
                'int',
                # 'float', 
                'float',
            ],
            'seed': 979,
        }),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'input_nested_keys': orm.List(list=[
            'angular_momentum_settings:s.rc',
            'angular_momentum_settings:s.qcut',
            'angular_momentum_settings:s.ncon',
            'angular_momentum_settings:s.nbas',
            # 'angular_momentum_settings:s.nproj',
            'angular_momentum_settings:s.debl',
            # 'local_potential_settings:rc(5)',
            # 'local_potential_settings:dvloc0',
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