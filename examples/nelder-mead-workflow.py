from aiida.engine import run_get_node
from aiida import orm

from aiida_opsp.workflows.ls import NelderMeadWorkChain
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

def run():
    code = orm.load_code('oncv4@localhost')

    conf_name = orm.Str('Li-low')
    angular_momentum_settings = orm.Dict(
        dict={
            's': {
                'nbas': 7,
                'nproj': 2,
            },
            'p': {
                'qcut': 7.3,
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
            'max_iter': 100,
            'xtol': 1e-3,
            'ftol': 1e-3,
        }),
        'init_individual': orm.Dict(dict={
            's_rc': 2.2,
            's_qcut': 5.0,
            's_ncon': 3,
            's_debl': 2.0,
            'p_rc': 2.2,
            'rc(5)': 2.2,
        }),
        'seed': orm.Int(1),
        'evaluate_process': OncvPseudoBaseWorkChain,
        'variable_info': orm.Dict(dict={
            's_rc': {
                'key_name': 'angular_momentum_settings:s.rc',
                'var_type': 'float',
                'space': {
                    'ref_to': 'rc(5)',
                    'range': [0.0, 1.0],
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
            'p_rc': {
                'key_name': 'angular_momentum_settings:p.rc',
                'var_type': 'float',
                'space': {
                    'ref_to': 'rc(5)',
                    'range': [0.0, 1.0],
                },
                'local_optimize': True,
            },
            's_ncon': {
                'key_name': 'angular_momentum_settings:s.ncon',
                'var_type': 'int',
                'space': {
                    'range': [3, 4],
                },
                'local_optimize': False,
            },
            's_debl': {
                'key_name': 'angular_momentum_settings:s.debl',
                'var_type': 'float',
                'space': {
                    'range': [0.0, 3.0],
                },
                'local_optimize': True,
            },
            'rc(5)': {
                'key_name': 'local_potential_settings:rc(5)',
                'var_type': 'float',
                'space': {
                    'range': [2.0, 3.0],
                },
                'group': 'local',
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
    res, node = run_get_node(NelderMeadWorkChain, **inputs)

    return res, node
    

if __name__ == '__main__':
    res, node = run()