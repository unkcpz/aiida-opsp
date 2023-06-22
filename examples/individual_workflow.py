from aiida import orm
from aiida.engine import run_get_node
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

conf_name = orm.Str('Ga')
angular_momentum_settings = orm.Dict(
    dict={
        's': {
            'nproj': 2,
        },
        'p': {
            'nproj': 2,
        }, 
        'd': {
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

def run():
    code = orm.load_code('oncv4@localhost')

    inputs = {
        'evaluate_process': OncvPseudoBaseWorkChain,
        'variable_info': orm.Dict(dict={
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
                    'high': 2.0,
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
        'fixture_inputs': {
            'code': code,
            'conf_name': conf_name,
            'lmax': orm.Int(2),
            'angular_momentum_settings': angular_momentum_settings,
            'local_potential_settings': local_potential_settings,
            'nlcc_settings': nlcc_settings,
            'run_atomic_test': orm.Bool(False),
            'dump_psps': orm.Bool(False),
        }
    }