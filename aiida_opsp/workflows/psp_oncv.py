from aiida.engine import WorkChain, submit
from aiida import orm
from aiida.engine import ToContext
from aiida_opsp.calcjob import OncvPseudoCalculation


class OncvPseudoBaseWorkChain(WorkChain):
    """Wrap of OncvPseudoCalculation calcjob
    calculate and output `results` as result_key for GA and add error handler"""
    
    @classmethod
    def define(cls, spec):
        super(OncvPseudoBaseWorkChain, cls).define(spec)

        spec.expose_inputs(OncvPseudoCalculation, exclude=['metadata'])
        spec.expose_outputs(OncvPseudoCalculation)
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
        inputs = self.exposed_inputs(OncvPseudoCalculation)
        # import ipdb; ipdb.set_trace()
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
        
        self.out_many(
            self.exposed_outputs(workchain, OncvPseudoCalculation)
        )
        
        # a very experiment way to define evaluate value for accuracy of psp.
        g_factor = 1000 # the factor (weight) for ground state error, we what the ground state described accurate 
        d = workchain.outputs.output_parameters
        self.report(d.get_dict())
        result = (d['tc_0']['state_error_avg'] * g_factor + \
                d['tc_1']['state_error_avg'] + \
                d['tc_2']['state_error_avg']) / 3
        
        # GA need use max for best results error close to 0 is max
        result = -abs(result)
        
        self.out('result', orm.Float(result).store())
        