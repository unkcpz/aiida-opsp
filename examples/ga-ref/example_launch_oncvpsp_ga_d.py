from aiida.engine import run_get_pk
from aiida import orm
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

code = orm.load_code('oncv4@localhost')
conf_name = orm.Str('Ga')

angular_momentum_settings = orm.Dict(
    dict={
        's': {
            'rc': 1.65,
            'qcut': 4.4,
            'ncon': 4,
            'nbas': 8,
            'nproj': 2,
            'debl': 3.0,
        },
        'p': {
            'rc': 1.75,
            'qcut': 5.8,
            'ncon': 4,
            'nbas': 8,
            'nproj': 2,
            'debl': 3.0,
        }, 
        'd': {
            'rc': 1.9,
            'qcut': 9.0,
            'ncon': 4,
            'nbas': 9,
            'nproj': 2,
            'debl': 2.0,
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
        'icmod': 3,
        'fcfact': 5.0,
        'rcfact': 1.4,
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