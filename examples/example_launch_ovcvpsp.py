from aiida.engine import run_get_pk
from aiida import orm
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

code = orm.load_code('oncv4@localhost')
conf_name = orm.Str('Li-low')

angular_momentum_settings = orm.Dict(
    dict={
        's': {
            'rc': 2.2,
            'qcut': 5.0,
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
        'dvloc0': 2.0,
    }
)
nlcc_settings = orm.Dict(
    dict={
        'icmod': 4,
        'fcfact': 5.0,
        'rcfact': 1.4,
    }
)

def run():
    inputs = {
        'code': code,
        'conf_name': conf_name,
        'lmax': orm.Int(1),
        'angular_momentum_settings': angular_momentum_settings,
        'local_potential_settings': local_potential_settings,
        'nlcc_settings': nlcc_settings,
        'run_atomic_test': orm.Bool(True),
        'dump_psp': orm.Bool(False),
        'weight_unbound': orm.Float(1.0),
        'fd_max': orm.Float(6.0),
    }
    res, pk = run_get_pk(OncvPseudoBaseWorkChain, **inputs)
    
    return res, pk

if __name__ == '__main__':
    res, pk = run()
    print(res['result'], pk)