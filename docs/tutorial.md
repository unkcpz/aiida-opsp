# Tutorial

!!! warning

    Use with caution. The plugin is still under development, and the tutorial is not complete yet.

## How to run pseudopotential generation

(TODO: on code side, I need to add support for customize electron configuration) Currently, we hard code the electron configuration of 

* `Ba_d` to be `[Kr] 5s2 5p6 6s2`.
* `Ga` to be `[Ar] 3d10 4s2 4p1`.
* `Li-low` to be `[He] 2s1`.
* `Li-high` to be `1s2 2s1`.
* `Na` to be `[He] 2s2 2p6 3s1`.

Under the hood, the plugin run workflow to call the `oncvpsp` code to generate the pseudopotential. 
The `oncvpsp` code should be installed and setup up for the plugin to work.
The `oncvpsp` code is installed in the docker image, so you can use the docker image to run the workflow.

Compare to run the code directly by provide raw input files, the plugin provides a general interface to run the pseudopotential generation workflow that can potentially be used for other pseudopotential generation codes. 
It also runs the post-processing workflow to extract the calculation results from the output files, provide the precision and convergence information. 

A typical script to run the pseudopotential generation workflow is as follows:

``` py title="psp_gen.py"
from aiida.engine import run_get_pk
from aiida import orm
from aiida_opsp.workflows.psp_oncv import OncvPseudoBaseWorkChain

code = orm.load_code('oncvpsp@localhost')
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
```

Run it within the docker container by:

```bash
verdi run psp_gen.py
```

## How to run optimization

There are two types of optimization workflow, the generic algorithm and the local optimization algorithm.
The local optimization algorithm is optional and can be used to further optimize the pseudopotential generated by the generic algorithm.

Here is a typical script to run the optimization workflow:

``` py title="psp_opt_sodium.py"
from aiida.engine import run_get_node, submit
from aiida import orm

from aiida_opsp.workflows.ga import GeneticAlgorithmWorkChain

conf_name = orm.Str('Na')
lmax = 1 # (1)
angular_momentum_settings = orm.Dict(
    dict={
        's': {
            'nproj': 2,
        },
        'p': {
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
        'icmod': 0,
        'fcfact': 0.00,
        #'rcfact': 1.4,
    }
)

variable_info = {
    's_rc': {
        'key_name': 'angular_momentum_settings:s.rc',
        'var_type': 'float',
        'space': {
            'ref_to': 'rc(5)',
            'range': [0, 2.0],
        },
        'group': 's',
        'local_optimize': True,
    },
    's_qcut': {
        'key_name': 'angular_momentum_settings:s.qcut',
        'var_type': 'float',
        'space': {
            'range': [4.0, 10.0],
        },
        'group': 's',
        'local_optimize': True,
    },
    's_ncon': {
        'key_name': 'angular_momentum_settings:s.ncon',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
        },
        'group': 's',
        'local_optimize': False,
    },
    's_nbas': {
        'key_name': 'angular_momentum_settings:s.nbas',
        'var_type': 'int',
        'space': {
            'ref_to': 's_ncon',
            'range': [3, 5],
        },
        'group': 's',
        # 'local_optimize': False,
    },
    's_debl': {
        'key_name': 'angular_momentum_settings:s.debl',
        'var_type': 'float',
        'space': {
            'range': [0, 5.0],
        },
        'group': 's',
        'local_optimize': True,
    },
    'p_rc': {
        'key_name': 'angular_momentum_settings:p.rc',
        'var_type': 'float',
        'space': {
            'ref_to': 'rc(5)',
            'range': [0, 2.0],
        },
        'group': 'p',
        'local_optimize': True,
    },
    'p_qcut': {
        'key_name': 'angular_momentum_settings:p.qcut',
        'var_type': 'float',
        'space': {
            'range': [4.0, 10.0],
        },
        'group': 'p',
        'local_optimize': True,
    },
    'p_ncon': {
        'key_name': 'angular_momentum_settings:p.ncon',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
        },
        'group': 'p',
        'local_optimize': False,
    },
    'p_nbas': {
        'key_name': 'angular_momentum_settings:p.nbas',
        'var_type': 'int',
        'space': {
            'range': [3, 5],
            'ref_to': 'p_ncon',
        },
        'group': 'p',
        # 'local_optimize': False,
    },
    'p_debl': {
        'key_name': 'angular_momentum_settings:p.debl',
        'var_type': 'float',
        'space': {
            'range': [0, 5.0],
        },
        'group': 'p',
        'local_optimize': True,
    },       
    'lpopt': {
        'key_name': 'local_potential_settings:lpopt',
        'var_type': 'int',
        'space': {
            'range': [1, 5],
        },
        'group': 'local',
        'local_optimize': False,
    },
    'rc(5)': {
        'key_name': 'local_potential_settings:rc(5)',
        'var_type': 'float',
        'space': {
            'range': [0.5, 3.0],
        },
        'group': 'local',
        'local_optimize': False,
    },       
    'dvloc0': {
        'key_name': 'local_potential_settings:dvloc0',
        'var_type': 'float',
        'space': {
            'range': [0.0, 3.0],
        },
        'group': 'local',
        'local_optimize': False,
    },    
}

def run(on_daemon=False):
    code = orm.load_code('oncvpsp@localhost')
    pw_code = orm.load_code('pw-7.2@localhost')

    inputs = {
        'ga_parameters': orm.Dict(dict={
            'seed': 2023,
            'num_generations': 10,
            'num_individuals': 20,
            'num_mating_individuals': 8,
            'num_elite_individuals': 2,
            'num_new_individuals': 4,
            'num_offspring_individuals': 4,
            'max_thebest_count': 10,
            'elite_individual_mutate_probability': 0.4,
            'mediocre_individual_mutate_probability': 0.8,
            'individual_generate_max_iteration': 40,
        }),
        'generate_evaluate_process': orm.Str('aiida.workflows:opsp.pseudo.oncvpsp'),
        'score_evaluate_process': orm.Str('aiida.workflows:opsp.verify.sssp'),
        'score_evaluate_parameters': {
            'upper_bound_ecutwfc': orm.Float(80.0),
            'sssp': {
                'code': pw_code,
                'configurations': orm.List(list=['XO', 'BCC']),
                'parallelization': orm.Dict(dict={'npool': 4}),
                'protocol': orm.Str('opsp'),
                'options': orm.Dict(dict={'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 36}, 'max_wallclock_seconds': 600}),
                'clean_workdir': orm.Bool(True),
            },
        },
        'variable_info': orm.Dict(dict=variable_info),
        'result_key': orm.Str('result'),
        'fixture_inputs': {
            'code': code,
            'conf_name': conf_name,
            'lmax': orm.Int(lmax),
            'angular_momentum_settings': angular_momentum_settings,
            'local_potential_settings': local_potential_settings,
            'nlcc_settings': nlcc_settings,
            'run_atomic_test': orm.Bool(True),
            'dump_psp': orm.Bool(False),
            'weight_unbound': orm.Float(0.5),
            'weight_ecut': orm.Float(0.0),
            'fd_max': orm.Float(10.0),
        },
        'local_optimization_process': orm.Str('aiida.workflows:opsp.optimize.nelder_mead'),
        'local_optimization_parameters': orm.Dict(dict={
            'evaluate_process': 'aiida.workflows:opsp.pseudo.oncvpsp',
            'max_iter': 10,
            'xtol': 1e-3, # this is absolute tolerance the presicion for the input parameters
            'ftol': 1e-3, # this is relative tolerance for the score
        }),
    }
    if on_daemon:
        node = submit(GeneticAlgorithmWorkChain, **inputs) 
        node.description = 'Na.optimized from dojo input'
        return None, node
    else:
        res, node = run_get_node(GeneticAlgorithmWorkChain, **inputs)
        node.description = 'Na.optimized from dojo input'
        return res, node

if __name__ == '__main__':
    res, pk = run(on_daemon=True)
    # print(res['result'].get_dict(), pk)
```

1. `lmax` maximum angular momentum for which psp is calculated (<=3)