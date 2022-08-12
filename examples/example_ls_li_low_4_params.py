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
                # 'rc': 2.2,
                # 'qcut': 7.0,
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
            'rc(5)': 1.1,
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
            'max_iter': 20,
            'xtol': 1e-1,
            'ftol': 1e-1,
            # 'init_simplex': create_init_simplex([1.5831, 5.5998, 2.2045, 9.5575], tol=0.1)  # fitness=183.25
            'init_simplex': create_init_simplex([2.8342, 4.4971, 2.5851, 8.5906], tol=0.1)  # fitness=16.13
            # 'init_simplex': [   # fitness=16.13
            #     [2.8342, 4.4971, 2.5851, 8.5906], 
            #     [2.9342, 4.3971, 2.6851, 9.5906], 
            #     [2.7342, 4.7971, 2.2851, 8.7906], 
            # ],
        }),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'input_nested_keys': orm.List(list=[
            'angular_momentum_settings:s.rc',
            'angular_momentum_settings:s.qcut',
            'angular_momentum_settings:p.rc',
            'angular_momentum_settings:p.qcut',
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
    res, pk = run_get_pk(LocalSearchWorkChain, **inputs)

    return res, pk

# def run_test():
#     inputs = {
#         'parameters': orm.Dict(dict={
#             'max_iter': 10,
#             'xtol': 1e-1,
#             'ftol': 1e-1,
#             'init_simplex': [[1.2, 0.9], [1.0, 2.0], [2.0, 1.0]],
#         }),
#         'evaluate_process': Rosenbrock,
#         'input_nested_keys': orm.List(list=[
#             'x',
#             'y',
#         ]
#         ),
#         'result_key': orm.Str('result'),
#         'fixture_inputs': {},
#     }
#     res, pk = run_get_pk(LocalSearchWorkChain, **inputs)

#     return res, pk
    

if __name__ == '__main__':
    res, pk = run()
    # res, pk = run_test()
    # print(res['result'].get_dict(), pk)