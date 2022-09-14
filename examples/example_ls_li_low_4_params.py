from aiida.engine import run_get_pk
from aiida.engine import calcfunction
from aiida import orm
import numpy as np

from aiida_opsp.workflows.ls import LocalSearchWorkChain, Rosenbrock, create_init_simplex
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

def run():
    code = orm.load_code('oncv4@localhost1')

    conf_name = orm.Str('Li-low')
    angular_momentum_settings = orm.Dict(
        dict={
            's': {
                # 'rc': 1.4,
                # 'qcut': 8.6,
                'ncon': 3,
                'nbas': 7,
                'nproj': 2,
                'debl': 2.0,
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
            'dvloc0': 0.0,
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
            'max_iter': 2,
            'xtol': 1e-1,
            'ftol': 1e-1,
            'init_vars': [2.872,  2.5731, 6.,     3.,     8.9284],
        }),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'vars_info': orm.Dict(dict={
            's_rc': {
                'key_name': 'angular_momentum_settings:s.rc',
                'type': 'float',
                'space': {
                    'low': 2.2, 
                    'high': 3.0,
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
                    'high': 4,
                },
                'local_optimize': False,
            },
            's_nbas': {
                'key_name': 'angular_momentum_settings:s.nbas',
                'type': 'int',
                'space': {
                    'low': 5, 
                    'high': 7,
                },
                # 'local_optimize': False,
            },
            # 'angular_momentum_settings:s.nproj',
            's_debl': {
                'key_name': 'angular_momentum_settings:s.debl',
                'type': 'float',
                'space': {
                    'low': 0, 
                    'high': 3.0,
                },
                'local_optimize': True,
            },
            # 'local_potential_settings:rc(5)',
            # 'local_potential_settings:dvloc0',            
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
    res, pk = run_get_pk(LocalSearchWorkChain, **inputs)

    return res, pk
    

if __name__ == '__main__':
    res, pk = run()
    # res, pk = run_test()
    # print(res['result'].get_dict(), pk)