from aiida.engine import WorkChain, submit
from aiida import orm
from aiida.engine import ToContext
from aiida_opsp.calcjob import OncvPseudoCalculation


class OncvPseudoWrapWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super(OncvPseudoWrapWorkChain, cls).define(spec)

        spec.input('parameters', valid_type=orm.Dict)    # rc_s, qcut_s
        spec.output('result', valid_type=orm.Float)

        spec.outline(
            cls.evaluate,
            cls.finalize,
        )
        spec.exit_code(201, 'ERROR_PARAMETERS_NOT_PROPER',
                    message='The oncv gereration failed because of bad parameters.')        
        spec.exit_code(301, 'ERROR_SUB_PROCESS_FAILED_ONCV',
                    message='The `ONCV` sub process failed.')

    def evaluate(self):
        # This is a bit improper: The new value should be created in a calculation.
        parameters = self.inputs.parameters.get_dict()
        rc_s = parameters['rc_s']
        qcut_s = parameters['qcut_s']
        
        inputs = get_inputs(rc_s, qcut_s)
        
        running = self.submit(OncvPseudoCalculation, **inputs)
        
        return ToContext(oncvwf=running)
        
    def finalize(self):
        workchain = self.ctx.oncvwf
        
        # only parse and set result when finish ok
        # it can be test configuration parse error that no test is done but psp is generated
        if not workchain.is_finished_ok:
            if 500 < workchain.exit_status < 600:
                self.report(
                    f"WF for oncv failed parameters not proper with exit status {workchain.exit_status}"
                )
                return self.exit_codes.ERROR_PARAMETERS_NOT_PROPER
            else:
                self.report(
                    f"WF for oncv failed with exit status {workchain.exit_status}"
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ONCV
        
        # a very experiment way to define evaluate value for accuracy of psp.
        g_factor = 1000 # the factor (weight) for ground state error, we what the ground state described accurate 
        d = output_parameters = workchain.outputs.output_parameters
        self.report(d.get_dict())
        result = (d['tc_0']['state_error_avg'] * g_factor + \
                d['tc_1']['state_error_avg'] + \
                d['tc_2']['state_error_avg']) / 3
        
        # GA need use max for best results error close to 0 is max
        result = -result
        
        self.out('result', orm.Float(result).store())
        
        
def get_inputs(rc_s, qcut_s):
    computer = orm.load_computer('localhost')
    code = orm.Code(
        label='oncv4-0', 
        remote_computer_exec=[computer, '/home/jyu/Projects/WP-OPSP/bin/oncvpsp.x'], 
        input_plugin_name='opsp.pseudo.oncv')

    conf_name = orm.Str('Li-s')
    angular_momentum_settings = orm.Dict(
        dict={
            's': {
                'rc': rc_s,
                'ncon': 4,
                'nbas': 8,
                'qcut': qcut_s,
                'nproj': 2,
                'debl': 1.0,
            },
            'p': {
                'rc': 1.1,
                'ncon': 4,
                'nbas': 8,
                'qcut': 9.0,
                'nproj': 2,
                'debl': 1.0,
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
            'icmod': 0,
            'fcfact': 0.25,
        }
    )
    inputs = {
        'code': code,
        'conf_name': conf_name,
        'lmax': orm.Int(1),
        'angular_momentum_settings': angular_momentum_settings,
        'local_potential_settings': local_potential_settings,
        'nlcc_settings': nlcc_settings,
        'run_atomic_test': orm.Bool(True),
        'dump_psp': orm.Bool(False),
        'metadata': {
            'options': {
                'resources': {
                    'num_machines': int(1)
                },
                'max_wallclock_seconds': int(60),
                'withmpi': False,
            },
            # 'dry_run':True,
        }
    }
    
    return inputs