from aiida.engine import run_get_pk
from aiida import orm
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

code = orm.load_code('oncv4@localhost')
conf_name = orm.Str('Ga')

angular_momentum_settings = orm.Dict(
    dict={
        's': {
            'rc': 1.85,
            'qcut': 4.6,
            'ncon': 3,
            'nbas': 7,
            'nproj': 2,
            'debl': 3.0,
        },
        'p': {
            'rc': 1.8,
            'qcut': 5.0,
            'ncon': 3,
            'nbas': 7,
            'nproj': 2,
            'debl': 3.0,
        }, 
        'd': {
            'rc': 2.6,
            'qcut': 4.6,
            'ncon': 3,
            'nbas': 7,
            'nproj': 2,
            'debl': 3.0,
        }, 
    }
)
local_potential_settings = orm.Dict(
    dict={
        'llcol': 4, # fix
        'lpopt': 5, # 1-5, algorithm enum set
        'rc(5)': 1.6,
        'dvloc0': 0.0,
    }
)
nlcc_settings = orm.Dict(
    dict={
        'icmod': 1,
        'fcfact': 0.5,
        'rcfact': 0.0,
    }
)


def run():
    inputs = {
        'code': code,
        'conf_name': conf_name,
        'lmax': orm.Int(2),
        'angular_momentum_settings': angular_momentum_settings,
        'local_potential_settings': local_potential_settings,
        'nlcc_settings': nlcc_settings,
        'run_atomic_test': orm.Bool(True),
        'dump_psp': orm.Bool(False),
    }
    res, pk = run_get_pk(OncvPseudoBaseWorkChain, **inputs)
    
    return res, pk

if __name__ == '__main__':
    res, pk = run()
    print(res['result'], pk)