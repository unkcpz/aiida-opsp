from aiida.engine import run_get_pk
from aiida import orm
from aiida.plugins import WorkflowFactory

SSSPVerificationWorkChain = WorkflowFactory('opsp.verify.sssp')

code = orm.load_code('oncv4@localhost')
pw_code = orm.load_code('pw-7.1@localhost')
#pw_code = orm.load_code('pw-7.0@daint-mc-mrcloud')
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
        'oncvpsp': {
            'code': code,
            'conf_name': conf_name,
            'lmax': orm.Int(1),
            'angular_momentum_settings': angular_momentum_settings,
            'local_potential_settings': local_potential_settings,
            'nlcc_settings': nlcc_settings,
        },
        'sssp': {
            'code': pw_code,
            'wavefunction_cutoff': orm.Float(30.0),
            'charge_density_cutoff': orm.Float(120.0),
            'protocol': orm.Str('test'),
            'configurations': orm.List(list=['XO', 'BCC']),
            'parallelization': orm.Dict(dict={'npool': 1}),
            'options': orm.Dict(dict={'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 3600}),
        }
    }
    res, pk = run_get_pk(SSSPVerificationWorkChain, **inputs)
    
    return res, pk

if __name__ == '__main__':
    res, pk = run()
    print(res['result'], pk)